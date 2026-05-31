"""Convert an FP32 ONNX core to FP16 at the ONNX level.

keep_io_types=True keeps stft_repr in / masks out as FP32 (cast nodes inserted
internally), so the separator feeds/receives fp32 unchanged — only the NN core
runs fp16. Output is repacked (small tensors inline) for onnxruntime path-load.

Run (no torch needed; needs an onnxruntime extra for the path-load check):
  uv run --group export --extra cpu python scripts/onnx/convert_fp16.py \
    --input tmp/models/model_core.onnx \
    --output tmp/models/model_core_fp16.onnx
"""

# ruff: noqa: T201 -- CLI build tool: stdout (progress/results) is its interface.

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxconverter_common import float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an ONNX model to FP16.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sanitize_fp16_initializers(model) -> int:
    """Zero non-finite fp16 weight values. The fp16 conversion turns rare fp32
    weight outliers (>65504) into inf, and a single inf weight makes its MatMul
    column NaN -> ~0.45% NaN in masks (one band, EP-independent — the op_block_list
    below does NOT catch this). The bad values sit among weights <=0.5, so zeroing
    the outliers is a negligible change. (Root-caused 2026-05-31, see converting.md.)"""
    fixed = 0
    for t in model.graph.initializer:
        if t.data_type != onnx.TensorProto.FLOAT16:
            continue
        a = numpy_helper.to_array(t).astype(np.float32)
        if not np.isfinite(a).all():
            fixed += int((~np.isfinite(a)).sum())
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
            t.CopyFrom(numpy_helper.from_array(a, t.name))
    return fixed


def save_fp16(model, output: Path) -> None:
    # onnx's external-data writer appends to an existing .data file; start clean.
    output.unlink(missing_ok=True)
    output.with_suffix(".onnx.data").unlink(missing_ok=True)
    onnx.save_model(
        model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output.name + ".data",
        size_threshold=4096,
        convert_attribute=True,
    )


# Keep numerically-sensitive ops in fp32 (fp16 NaN sources): Softmax (attention
# logit overflow) and the RMSNorm chain ReduceL2/Clip/Div (x/‖x‖ with eps=1e-12
# that underflows to 0 in fp16 -> div-by-zero). These are weightless, so blocking
# them keeps the size win (Linear weights stay fp16). Extend the lib default.
EXTRA_BLOCK = ["Softmax", "ReduceL2", "Clip", "Div"]
OP_BLOCK = list(dict.fromkeys([*float16.DEFAULT_OP_BLOCK_LIST, *EXTRA_BLOCK]))


def mb(p: Path) -> float:
    return (p.stat().st_size + p.with_suffix(".onnx.data").stat().st_size) / 1e6


args = parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

print(f"loading {args.input.name} (FP32, {mb(args.input):.0f} MB) ...")
model = onnx.load(str(args.input))

print(f"converting to FP16 (keep_io_types=True, block={EXTRA_BLOCK}) ...")
model16 = float16.convert_float_to_float16(
    model, keep_io_types=True, disable_shape_infer=True, op_block_list=OP_BLOCK
)
del model

print("saving (small tensors inline, weights external) ...")
save_fp16(model16, args.output)

# Post-save sanitize: the conversion materializes the fp32 weight outliers as inf
# only in the saved external initializers, so reload and zero them, then re-save.
reloaded = onnx.load(str(args.output))
nfix = sanitize_fp16_initializers(reloaded)
if nfix:
    save_fp16(reloaded, args.output)
print(f"  sanitized {nfix} non-finite fp16 weight value(s) -> 0")
print(f"  wrote {args.output.name} (+data, {mb(args.output):.0f} MB)")

print("verifying ORT path-load ...")
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(
    str(args.output), sess_options=so, providers=["CPUExecutionProvider"]
)
print("OK. inputs:", [(i.name, i.type, i.shape) for i in sess.get_inputs()])
print("    outputs:", [(o.name, o.type, o.shape) for o in sess.get_outputs()])
