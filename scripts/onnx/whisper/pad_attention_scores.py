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

The attention score is `MatMul(Q, Transpose(K, perm=[0, 2, 1]))`. This appends
one zero column to `Q` and one zero row to the transposed key, so the product
runs with `K = 65`:

    scores = MatMul(Pad(Q, [0, 0, 0, 0, 0, 1]), Pad(Kt, [0, 0, 0, 0, 1, 0]))

65 is never a multiple of four, so the provider takes its scalar kernel, which
is exact on the same device. The added row and column multiply to zero, so every
score is the number it was, and the output keeps its shape: no slice afterwards.

Both paddings are constants, which is the point. Padding the key length instead
would have to read that length at run time (`Shape -> Gather -> Mod -> Equal`),
and shape arithmetic runs on the CPU provider: it splits the step's GPU work into
pieces and forces a synchronisation around each attention. Measured on
`decoder_model_merged_fp16` at 30 s of audio, the run-time form cost 64.1 ms per
token on a desktop NVIDIA against 33.5 ms without any padding, and added 14 queue
submissions per token; the head dimension is known at export and costs none of
that.

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
SCORE_RANK = 3
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
    """Widen the head dimension of each attention score product by one."""
    targets = attention_scores(graph)
    if not targets:
        return 0
    prefix = f"{PAD_PREFIX}/{index}"
    query_pads = np.zeros(2 * SCORE_RANK, dtype=np.int64)
    query_pads[-1] = 1
    key_pads = np.zeros(2 * SCORE_RANK, dtype=np.int64)
    key_pads[-2] = 1
    for name, value in (("query_pads", query_pads), ("key_pads", key_pads)):
        graph.initializer.append(numpy_helper.from_array(value, f"{prefix}/{name}"))

    rewritten = []
    for node in graph.node:
        if node not in targets:
            rewritten.append(node)
            continue
        base = f"{prefix}/{node.name or node.output[0]}"
        query = f"{base}/query"
        keys = f"{base}/keys"
        rewritten.extend(
            [
                helper.make_node(
                    "Pad",
                    [node.input[0], f"{prefix}/query_pads"],
                    [query],
                    name=query,
                    mode="constant",
                ),
                helper.make_node(
                    "Pad",
                    [node.input[1], f"{prefix}/key_pads"],
                    [keys],
                    name=keys,
                    mode="constant",
                ),
                helper.make_node(
                    "MatMul", [query, keys], [node.output[0]], name=node.name
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
            print(f"graph {index} ({graph.name}): {count} attention scores widened")
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
