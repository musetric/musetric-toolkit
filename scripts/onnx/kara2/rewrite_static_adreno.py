# ruff: noqa: T201
"""Static rewrite of UVR-MDX-NET KARA2 that avoids the Adreno 6xx broken kernels.

On Adreno 660 the onnxruntime-web WebGPU kernels for BatchNormalization, Conv
and ConvTranspose corrupt this graph nondeterministically; Transpose, MatMul,
Relu, Add and Mul compute it correctly. The rewrite keeps only the working
operators:

- BatchNormalization becomes Mul + Add with folded running statistics.
- Conv becomes a sum over kernel offsets. Each offset slices the padded input
  with the stride, flattens it to [C_in, H * W] and multiplies it by the
  [C_out, C_in] weight slice on the left, so no transposition appears.
- ConvTranspose with a non-overlapping kernel (kernel == stride) becomes one
  weight-left MatMul per kernel offset; the offsets are interleaved into the
  upsampled plane with Concat + Reshape.

The graph is pinned to batch 1 and the input shape of the source graph.

Usage:
    uv run python scripts/onnx/kara2/rewrite_static_adreno.py \
        --model UVR_MDXNET_KARA_2.onnx --out kara2_adreno.onnx --check
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

BIAS_INPUT = 2
BATCHNORM_INPUTS = 5
BATCHNORM_EPSILON = 1e-5
CHECK_TOLERANCE = 1e-3


@dataclass
class RewriteContext:
    shapes: dict[str, list[int]]
    initializers: dict[str, onnx.TensorProto]
    nodes: list[onnx.NodeProto] = field(default_factory=list)
    constants: list[onnx.TensorProto] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def constant(self, name: str, values: np.ndarray) -> str:
        self.constants.append(numpy_helper.from_array(values, name))
        return name

    def shape(self, name: str, dims: list[int]) -> str:
        return self.constant(name, np.array(dims, dtype=np.int64))

    def node(self, op: str, inputs: list[str], output: str, **attrs: object) -> str:
        self.nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))
        return output

    def weight(self, name: str) -> np.ndarray:
        return numpy_helper.to_array(self.initializers[name]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="source KARA2 .onnx")
    parser.add_argument("--out", required=True, help="rewritten .onnx output")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare source and rewrite on the CPU provider",
    )
    return parser.parse_args()


def pin_batch(model: onnx.ModelProto) -> None:
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[0].dim_value = 1


def static_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = shape_inference.infer_shapes(model).graph
    values = [*inferred.input, *inferred.value_info, *inferred.output]
    return {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in values
    }


def attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def rewrite_batchnorm(ctx: RewriteContext, node: onnx.NodeProto) -> None:
    x_name, scale_name, bias_name, mean_name, var_name = node.input[:BATCHNORM_INPUTS]
    epsilon = float(attributes(node).get("epsilon", BATCHNORM_EPSILON))
    inverse = ctx.weight(scale_name) / np.sqrt(ctx.weight(var_name) + epsilon)
    shift = ctx.weight(bias_name) - ctx.weight(mean_name) * inverse
    broadcast = [1, inverse.size, 1, 1]
    scale = ctx.constant(f"{node.name}/scale", inverse.reshape(broadcast))
    offset = ctx.constant(f"{node.name}/shift", shift.reshape(broadcast))
    scaled = ctx.node("Mul", [x_name, scale], f"{node.name}/mul")
    ctx.node("Add", [scaled, offset], node.output[0])


def rewrite_conv(ctx: RewriteContext, node: onnx.NodeProto) -> None:
    weight = ctx.weight(node.input[1])
    c_out, c_in, kernel_h, kernel_w = weight.shape
    _, _, height, width = ctx.shapes[node.input[0]]
    attrs = attributes(node)
    pad_top, pad_left, pad_bottom, pad_right = attrs.get("pads", [0, 0, 0, 0])
    stride_h, stride_w = attrs.get("strides", [1, 1])
    padded_h = height + pad_top + pad_bottom
    padded_w = width + pad_left + pad_right
    out_h = (padded_h - kernel_h) // stride_h + 1
    out_w = (padded_w - kernel_w) // stride_w + 1

    source = node.input[0]
    if pad_top or pad_left or pad_bottom or pad_right:
        pads = ctx.shape(
            f"{node.name}/pads", [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right]
        )
        zero = ctx.constant(f"{node.name}/zero", np.array([0.0], dtype=np.float32))
        source = ctx.node(
            "Pad", [source, pads, zero], f"{node.name}/padded", mode="constant"
        )

    flat_shape = ctx.shape(f"{node.name}/flat_shape", [c_in, out_h * out_w])
    whole_plane = (
        kernel_h == 1
        and kernel_w == 1
        and stride_h == 1
        and stride_w == 1
        and (padded_h, padded_w) == (out_h, out_w)
    )
    summed = ""
    for row in range(kernel_h):
        for col in range(kernel_w):
            prefix = f"{node.name}/k{row}{col}"
            patch = source
            if not whole_plane:
                starts = ctx.shape(f"{prefix}/starts", [row, col])
                ends = ctx.shape(
                    f"{prefix}/ends",
                    [
                        row + stride_h * (out_h - 1) + 1,
                        col + stride_w * (out_w - 1) + 1,
                    ],
                )
                axes = ctx.shape(f"{prefix}/axes", [2, 3])
                steps = ctx.shape(f"{prefix}/steps", [stride_h, stride_w])
                patch = ctx.node(
                    "Slice", [source, starts, ends, axes, steps], f"{prefix}/slice"
                )
            flat = ctx.node("Reshape", [patch, flat_shape], f"{prefix}/flat")
            kernel = ctx.constant(
                f"{prefix}/w", np.ascontiguousarray(weight[:, :, row, col])
            )
            product = ctx.node("MatMul", [kernel, flat], f"{prefix}/mm")
            summed = (
                ctx.node("Add", [summed, product], f"{prefix}/sum")
                if summed
                else product
            )

    bias = ctx.constant(
        f"{node.name}/bias", ctx.weight(node.input[BIAS_INPUT]).reshape(c_out, 1)
    )
    biased = ctx.node("Add", [summed, bias], f"{node.name}/biased")
    out_shape = ctx.shape(f"{node.name}/out_shape", [1, c_out, out_h, out_w])
    ctx.node("Reshape", [biased, out_shape], node.output[0])


def rewrite_conv_transpose(ctx: RewriteContext, node: onnx.NodeProto) -> None:
    weight = ctx.weight(node.input[1])
    c_in, c_out, kernel_h, kernel_w = weight.shape
    _, _, height, width = ctx.shapes[node.input[0]]
    attrs = attributes(node)
    if (
        list(attrs.get("strides", [1, 1])) != [kernel_h, kernel_w]
        or any(attrs.get("pads", [0, 0, 0, 0]))
        or attrs.get("group", 1) != 1
    ):
        message = f"{node.name}: only non-overlapping ConvTranspose is supported"
        raise ValueError(message)

    flat_shape = ctx.shape(f"{node.name}/flat_shape", [c_in, height * width])
    flat = ctx.node("Reshape", [node.input[0], flat_shape], f"{node.name}/flat")
    cell_shape = ctx.shape(f"{node.name}/cell_shape", [c_out, height, width, 1])
    row_shape = ctx.shape(
        f"{node.name}/row_shape", [c_out, height, 1, width * kernel_w]
    )
    rows = []
    for row in range(kernel_h):
        cells = []
        for col in range(kernel_w):
            prefix = f"{node.name}/k{row}{col}"
            kernel = ctx.constant(
                f"{prefix}/w", np.ascontiguousarray(weight[:, :, row, col].T)
            )
            product = ctx.node("MatMul", [kernel, flat], f"{prefix}/mm")
            cells.append(ctx.node("Reshape", [product, cell_shape], f"{prefix}/cell"))
        joined = ctx.node("Concat", cells, f"{node.name}/r{row}/cells", axis=3)
        rows.append(ctx.node("Reshape", [joined, row_shape], f"{node.name}/r{row}"))
    plane = ctx.node("Concat", rows, f"{node.name}/rows", axis=2)
    plane_shape = ctx.shape(
        f"{node.name}/plane_shape", [1, c_out, height * kernel_h, width * kernel_w]
    )
    upsampled = ctx.node("Reshape", [plane, plane_shape], f"{node.name}/plane")
    bias = ctx.constant(
        f"{node.name}/bias",
        ctx.weight(node.input[BIAS_INPUT]).reshape(1, c_out, 1, 1),
    )
    ctx.node("Add", [upsampled, bias], node.output[0])


REWRITES = {
    "BatchNormalization": rewrite_batchnorm,
    "Conv": rewrite_conv,
    "ConvTranspose": rewrite_conv_transpose,
}


def rewrite(model: onnx.ModelProto) -> onnx.ModelProto:
    pin_batch(model)
    graph = model.graph
    ctx = RewriteContext(
        shapes=static_shapes(model),
        initializers={init.name: init for init in graph.initializer},
    )
    for node in graph.node:
        handler = REWRITES.get(node.op_type)
        if handler is None:
            ctx.nodes.append(node)
            continue
        handler(ctx, node)
        ctx.counts[node.op_type] = ctx.counts.get(node.op_type, 0) + 1

    used = {name for node in ctx.nodes for name in node.input}
    kept = [init for init in graph.initializer if init.name in used]
    rewritten = helper.make_graph(
        ctx.nodes,
        graph.name,
        list(graph.input),
        list(graph.output),
        initializer=kept + ctx.constants,
    )
    result = helper.make_model(
        rewritten, opset_imports=model.opset_import, producer_name=__name__
    )
    result.ir_version = model.ir_version
    onnx.checker.check_model(result, full_check=True)
    print("rewritten:", ctx.counts)
    return result


def check(source_path: Path, rewrite_path: Path, shape: list[int]) -> None:
    import onnxruntime as ort  # noqa: PLC0415

    count = int(np.prod(shape))
    values = ((np.arange(count) * 37 + 11) % 211) / 211 - 0.5
    feed = values.astype(np.float32).reshape(shape)
    outputs = []
    for path in (source_path, rewrite_path):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        outputs.append(session.run(None, {name: feed})[0])
    difference = float(np.max(np.abs(outputs[0] - outputs[1])))
    print(f"cpu max abs difference {difference:.3e}")
    if difference > CHECK_TOLERANCE:
        message = f"rewrite differs from the source by {difference}"
        raise SystemExit(message)


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model)
    shape = [
        1,
        *[dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim[1:]],
    ]
    result = rewrite(model)
    onnx.save(result, args.out)
    size = Path(args.out).stat().st_size / 1e6
    print(f"saved {args.out}: {len(result.graph.node)} nodes, {size:.1f} MB")
    if args.check:
        check(Path(args.model), Path(args.out), shape)


if __name__ == "__main__":
    main()
