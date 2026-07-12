"""Feasibility probe for the fusion (level-C) track: emit tiny ONNX graphs that
use the ORT contrib fused ops we want (SimplifiedLayerNormalization, FastGelu,
SkipSimplifiedLayerNormalization), so a Node-side runner can check whether the
onnxruntime-node WebGPU EP actually executes them as WebGPU kernels (vs CPU
fallback).
"""

# ruff: noqa: T201

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MS = "com.microsoft"
DIM = 384


def _init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr.astype(np.float32), name)


def simplified_layernorm() -> onnx.ModelProto:
    # X[1,N,DIM] fp16, scale[DIM] fp16 -> Y[1,N,DIM]
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, "N", DIM])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "N", DIM])
    scale = _init("scale", np.ones(DIM))
    node = helper.make_node(
        "SimplifiedLayerNormalization",
        ["X", "scale"],
        ["Y"],
        domain=MS,
        axis=-1,
        epsilon=1e-6,
        stash_type=1,
    )
    graph = helper.make_graph([node], "slnorm", [x], [y], [scale])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(MS, 1)],
    )


def fastgelu() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, "N", DIM])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "N", DIM])
    bias = _init("bias", np.zeros(DIM))
    node = helper.make_node("FastGelu", ["X", "bias"], ["Y"], domain=MS)
    graph = helper.make_graph([node], "fastgelu", [x], [y], [bias])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(MS, 1)],
    )


def skip_simplified_layernorm() -> onnx.ModelProto:
    # input + skip -> SimplifiedLayerNorm. Fuses the residual add too.
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, "N", DIM])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "N", DIM])
    scale = _init("scale", np.ones(DIM))
    # zero skip as a [DIM] initializer -> tests whether ORT broadcasts skip
    # (so skip-less RMSNorm can fuse to SkipSimplifiedLayerNorm without a full
    # [1,N,DIM] zero tensor).
    skip = _init("skip", np.zeros(DIM))
    node = helper.make_node(
        "SkipSimplifiedLayerNormalization",
        ["X", "skip", "scale"],
        ["Y", "", "", "Out"],
        domain=MS,
        epsilon=1e-6,
    )
    out = helper.make_tensor_value_info("Out", TensorProto.FLOAT, [1, "N", DIM])
    graph = helper.make_graph([node], "skipslnorm", [x], [y, out], [scale, skip])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(MS, 1)],
    )


def onnx_gelu_erf() -> onnx.ModelProto:
    # standard ONNX Gelu (opset 20), approximate="none" = exact erf GELU.
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, "N", DIM])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "N", DIM])
    node = helper.make_node("Gelu", ["X"], ["Y"], approximate="none")
    graph = helper.make_graph([node], "gelu_erf", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


def bias_gelu_erf() -> onnx.ModelProto:
    # com.microsoft BiasGelu = erf GELU of (X + bias); zero bias = exact GELU.
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, "N", DIM])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "N", DIM])
    bias = _init("bias", np.zeros(DIM))
    node = helper.make_node("BiasGelu", ["X", "bias"], ["Y"], domain=MS)
    graph = helper.make_graph([node], "biasgelu", [x], [y], [bias])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(MS, 1)],
    )


def rms_normalization() -> onnx.ModelProto:
    # Standard ONNX RMSNormalization (opset 23), onnx domain — plain RMSNorm, no
    # skip. If the WebGPU EP runs it, every RMSNorm fuses 1:1 with no residual
    # detection.
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, "N", DIM])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "N", DIM])
    scale = _init("scale", np.ones(DIM))
    node = helper.make_node(
        "RMSNormalization", ["X", "scale"], ["Y"], axis=-1, epsilon=1e-6
    )
    graph = helper.make_graph([node], "rmsnorm", [x], [y], [scale])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("tmp/probe"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in [
        ("slnorm", simplified_layernorm),
        ("fastgelu", fastgelu),
        ("skipslnorm", skip_simplified_layernorm),
        ("rmsnorm", rms_normalization),
        ("gelu_erf", onnx_gelu_erf),
        ("biasgelu", bias_gelu_erf),
    ]:
        model = fn()
        path = args.out_dir / f"probe_{name}.onnx"
        onnx.save_model(model, str(path))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
