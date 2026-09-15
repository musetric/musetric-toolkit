"""Keep the decoder's attention scores off the WebGPU MatMul kernel Adreno 600 breaks.

The WebGPU execution provider multiplies `A [batch, M, K] x B [batch, K, N]` with
a packed vec4 shader whenever both `K` and `N` are multiples of four. On Adreno
600-series adapters that shader returns wrong values when `M` is small, as it is
in a decoder (one query row per cached step, the prompt on the first step): with
`K = 64` and `N` the key length, every `N` that is a multiple of four but not of
32 comes back as garbage, in float16 and float32 alike. Self-attention hits it
on every fourth step as the cache grows; cross-attention hits it on every step,
because the key length is 1500. The corrupted hidden state then feeds the next
layers' keys and values, so the cache carries the error to every later step.

The attention score is `MatMul(Q, Transpose(K, perm=[0, 2, 1]))`. When the key
length is a multiple of four, this appends one zero key column before the
product and slices it off after:

    n      = Shape(Kt)[2]
    pad    = Cast(Mod(n, 4) == 0)
    scores = Slice(MatMul(Q, Pad(Kt, [0, 0, 0, 0, 0, pad])), 0:n on axis 2)

`N + 1` is never a multiple of four, so the provider takes its scalar kernel,
which is exact on the same device. The zero column produces a zero score that
the slice drops, so the result is identical to the original product; the extra
column costs one key per step.

    uv run python scripts/onnx/whisper/pad_attention_scores.py \
      --input  decoder_model_merged_fp16.onnx \
      --output decoder_model_merged_fp16.onnx
"""

# ruff: noqa: T201

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

KEY_TRANSPOSE = [0, 2, 1]
KEY_AXIS = 2
PACKED_MULTIPLE = 4
PAD_PREFIX = "pad_attention_scores"


def subgraphs(graph):
    """The graph itself and every graph nested in its nodes' attributes."""
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from subgraphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from subgraphs(nested)


def attention_scores(graph):
    """MatMul nodes whose right operand is a key tensor transposed to [batch, K, N]."""
    producers = {output: node for node in graph.node for output in node.output}
    found = []
    for node in graph.node:
        if node.op_type != "MatMul" or len(node.input) != 2:  # noqa: PLR2004
            continue
        producer = producers.get(node.input[1])
        if producer is None or producer.op_type != "Transpose":
            continue
        perm = next(
            (list(a.ints) for a in producer.attribute if a.name == "perm"), None
        )
        if perm == KEY_TRANSPOSE:
            found.append(node)
    return found


def count_affected(model):
    """Attention score products that still reach the packed kernel."""
    return sum(len(attention_scores(graph)) for graph in subgraphs(model.graph))


def rewrite(graph, index):
    """Pad the key length past a multiple of four around each attention score."""
    targets = attention_scores(graph)
    if not targets:
        return 0
    prefix = f"{PAD_PREFIX}/{index}"
    constants = {
        "axis": np.array([KEY_AXIS], dtype=np.int64),
        "multiple": np.array(PACKED_MULTIPLE, dtype=np.int64),
        "zero": np.array(0, dtype=np.int64),
        "pads_head": np.array([0, 0, 0, 0, 0], dtype=np.int64),
        "start": np.array([0], dtype=np.int64),
    }
    for name, value in constants.items():
        graph.initializer.append(numpy_helper.from_array(value, f"{prefix}/{name}"))

    rewritten = []
    for node in graph.node:
        if node not in targets:
            rewritten.append(node)
            continue
        base = f"{prefix}/{node.name or node.output[0]}"
        keys = node.input[1]
        shape = f"{base}/shape"
        length = f"{base}/length"
        pad = f"{base}/pad"
        padded = f"{base}/padded"
        product = f"{base}/product"
        rewritten.extend(
            [
                helper.make_node("Shape", [keys], [shape], name=shape),
                helper.make_node(
                    "Gather",
                    [shape, f"{prefix}/axis"],
                    [f"{length}_1d"],
                    name=length,
                    axis=0,
                ),
                helper.make_node(
                    "Squeeze",
                    [f"{length}_1d", f"{prefix}/start"],
                    [length],
                    name=f"{length}/squeeze",
                ),
                helper.make_node(
                    "Mod",
                    [length, f"{prefix}/multiple"],
                    [f"{pad}/mod"],
                    name=f"{pad}/mod",
                ),
                helper.make_node(
                    "Equal",
                    [f"{pad}/mod", f"{prefix}/zero"],
                    [f"{pad}/packed"],
                    name=f"{pad}/packed",
                ),
                helper.make_node(
                    "Cast",
                    [f"{pad}/packed"],
                    [f"{pad}/count"],
                    name=f"{pad}/count",
                    to=onnx.TensorProto.INT64,
                ),
                helper.make_node(
                    "Unsqueeze",
                    [f"{pad}/count", f"{prefix}/start"],
                    [f"{pad}/tail"],
                    name=f"{pad}/tail",
                ),
                helper.make_node(
                    "Concat",
                    [f"{prefix}/pads_head", f"{pad}/tail"],
                    [f"{pad}/pads"],
                    name=f"{pad}/pads",
                    axis=0,
                ),
                helper.make_node(
                    "Pad", [keys, f"{pad}/pads"], [padded], name=padded, mode="constant"
                ),
                helper.make_node(
                    "MatMul", [node.input[0], padded], [product], name=node.name
                ),
                helper.make_node(
                    "Unsqueeze",
                    [length, f"{prefix}/start"],
                    [f"{length}_end"],
                    name=f"{length}/end",
                ),
                helper.make_node(
                    "Slice",
                    [product, f"{prefix}/start", f"{length}_end", f"{prefix}/axis"],
                    [node.output[0]],
                    name=f"{base}/slice",
                ),
            ]
        )
    del graph.node[:]
    graph.node.extend(rewritten)
    return len(targets)


def apply(model):
    total = 0
    for index, graph in enumerate(list(subgraphs(model.graph))):
        count = rewrite(graph, index)
        if count:
            print(f"graph {index} ({graph.name}): {count} attention scores padded")
        total += count
    if not total:
        print("no attention score reaches the packed kernel; nothing to do")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    apply(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
