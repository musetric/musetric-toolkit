"""Re-export a MelBand ONNX core at a smaller static chunk T.
T frames ~= T/100 seconds. On a 6 GB GPU T=501 (~5s) avoids WDDM paging and runs
~25x faster per chunk than T=1101; bigger T needs more VRAM. See converting.md.

Pipeline (parameterized by T): ckpt -> export net_forward at (1,2050,T,2)
(dynamo) -> fp16 (op_block_list) + sanitize non-finite weights -> WebGPU patch
(Concat/Split trees <=15, optional fp16-softmax) -> fold constant rotary
Sin/Cos tables and prune dead initializers -> fixed-T ONNX artifact.
"""

# ruff: noqa: T201 -- CLI build tool: stdout (progress/results) is its interface.

import argparse
import contextlib
import gc
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

from musetric_toolkit.separate_audio.roformer.attend import Attend
from musetric_toolkit.separate_audio.roformer.mel_band_roformer import MelBandRoformer
from musetric_toolkit.separate_audio.roformer_utils import dict_to_namespace

GROUP = 15  # <=15 variable buffers -> <=16 storage buffers per WebGPU shader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a fixed-T FP16 ONNX core.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--softmax", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--raw-output", type=Path)
    return parser.parse_args()


# C901: single-shot graph-rewrite orchestration; splitting wouldn't aid clarity.
def split_wide_concat_split(graph):  # noqa: C901
    """Rewrite >15-wide Concat (inputs) / Split (outputs) into <=15-wide trees
    (associative, bit-identical) so no shader exceeds maxStorageBuffersPerShaderStage.
    """
    # late import: onnx loaded only when patching
    from onnx import helper, numpy_helper  # noqa: PLC0415

    inits = {t.name: t for t in graph.initializer}
    new_nodes, new_inits = [], []

    def chunks(seq):
        for s in range(0, len(seq), GROUP):
            yield seq[s : s + GROUP]

    def axis_of(node):
        return next((a.i for a in node.attribute if a.name == "axis"), 0)

    n_c = n_s = 0
    for node in graph.node:
        if node.op_type == "Concat" and len(node.input) > GROUP:
            n_c += 1
            ax, base = axis_of(node), node.name or f"concat_{n_c}"
            groups = []
            for gi, ch in enumerate(chunks(list(node.input))):
                if len(ch) == 1:
                    groups.append(ch[0])
                    continue
                out = f"{base}__g{gi}"
                new_nodes.append(
                    helper.make_node(
                        "Concat", ch, [out], name=f"{base}__g{gi}_n", axis=ax
                    )
                )
                groups.append(out)
            new_nodes.append(
                helper.make_node(
                    "Concat", groups, list(node.output), name=base, axis=ax
                )
            )
        elif node.op_type == "Split" and len(node.output) > GROUP:
            n_s += 1
            ax, base = axis_of(node), node.name or f"split_{n_s}"
            sizes = numpy_helper.to_array(inits[node.input[1]]).astype(np.int64)
            og = list(chunks(list(node.output)))
            sg = list(chunks(sizes.tolist()))
            names = [g[0] if len(g) == 1 else f"{base}__g{i}" for i, g in enumerate(og)]
            l1 = f"{base}__l1_sizes"
            new_inits.append(
                numpy_helper.from_array(
                    np.array([sum(s) for s in sg], dtype=np.int64), l1
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Split", [node.input[0], l1], names, name=f"{base}__l1", axis=ax
                )
            )
            for i, (g, s) in enumerate(zip(og, sg, strict=False)):
                if len(g) == 1:
                    continue
                sn = f"{base}__g{i}_sizes"
                new_inits.append(
                    numpy_helper.from_array(np.array(s, dtype=np.int64), sn)
                )
                new_nodes.append(
                    helper.make_node(
                        "Split", [names[i], sn], g, name=f"{base}__g{i}", axis=ax
                    )
                )
        else:
            new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    return n_c, n_s


def sanitize_fp16_initializers(model):
    """Zero non-finite fp16 weight values. The fp16 conversion turns rare fp32
    weight outliers (>65504) into inf, and a single inf weight makes its MatMul
    column NaN -> ~0.45% NaN in masks (one band, EP-independent). The offending
    weights sit among values <=0.5, so zeroing the outliers is a negligible change."""
    # late import: onnx loaded only when sanitizing
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    fixed = 0
    for t in model.graph.initializer:
        if t.data_type != onnx.TensorProto.FLOAT16:
            continue
        a = numpy_helper.to_array(t).astype(np.float32)
        bad = ~np.isfinite(a)
        if bad.any():
            fixed += int(bad.sum())
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
            t.CopyFrom(numpy_helper.from_array(a, t.name))
    return fixed


def unwrap_softmax_fp16(graph):
    """Drop fp32 cast wrap around each Softmax (keeps scores fp16 -> fits 2GB cap)."""
    producer = {o: n for n in graph.node for o in n.output}
    cons: dict = {}
    for n in graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    drop, names = set(), set()
    for sm in [n for n in graph.node if n.op_type == "Softmax"]:
        c1 = producer.get(sm.input[0])
        if (
            c1 is not None
            and c1.op_type == "Cast"
            and len(cons.get(c1.output[0], [])) == 1
        ):
            drop.add(id(c1))
            names.add(c1.output[0])
            sm.input[0] = c1.input[0]
        c2 = cons.get(sm.output[0], [])
        if len(c2) == 1 and c2[0].op_type == "Cast":
            drop.add(id(c2[0]))
            names.add(sm.output[0])
            sm.output[0] = c2[0].output[0]
    kept = [n for n in graph.node if id(n) not in drop]
    del graph.node[:]
    graph.node.extend(kept)
    for vi in [v for v in graph.value_info if v.name in names]:
        graph.value_info.remove(vi)
    return len(drop)


SLICE_STARTS_INPUT = 1
SLICE_ENDS_INPUT = 2
SLICE_AXES_INPUT = 3
SLICE_STEPS_INPUT = 4


def initializer_array(name, inits):
    # late import: onnx helpers loaded only when patching
    from onnx import numpy_helper  # noqa: PLC0415

    tensor = inits.get(name)
    return None if tensor is None else numpy_helper.to_array(tensor)


def constant_slice(node, inits):
    if node.op_type != "Slice" or not node.input:
        return None
    data = initializer_array(node.input[0], inits)
    starts = initializer_array(node.input[SLICE_STARTS_INPUT], inits)
    ends = initializer_array(node.input[SLICE_ENDS_INPUT], inits)
    if data is None or starts is None or ends is None:
        return None
    axes = (
        initializer_array(node.input[SLICE_AXES_INPUT], inits)
        if len(node.input) > SLICE_AXES_INPUT and node.input[SLICE_AXES_INPUT]
        else np.arange(len(starts), dtype=np.int64)
    )
    steps = (
        initializer_array(node.input[SLICE_STEPS_INPUT], inits)
        if len(node.input) > SLICE_STEPS_INPUT and node.input[SLICE_STEPS_INPUT]
        else np.ones(len(starts), dtype=np.int64)
    )
    slices = [slice(None)] * data.ndim
    for start, end, axis_value, step in zip(starts, ends, axes, steps, strict=False):
        axis = int(axis_value)
        slices[axis] = slice(int(start), int(end), int(step))
    return data[tuple(slices)]


def replace_nodes(graph, drop_ids):
    kept = [node for node in graph.node if id(node) not in drop_ids]
    del graph.node[:]
    graph.node.extend(kept)


def fold_constant_trig_nodes(graph):
    from onnx import numpy_helper  # noqa: PLC0415

    inits = {t.name: t for t in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}
    drop_ids = set()
    folded = 0
    for node in list(graph.node):
        if node.op_type not in {"Sin", "Cos"} or len(node.input) != 1:
            continue
        source = initializer_array(node.input[0], inits)
        source_producer = producer.get(node.input[0])
        if source is None and source_producer is not None:
            source = constant_slice(source_producer, inits)
        if source is None:
            continue
        op = np.sin if node.op_type == "Sin" else np.cos
        value = op(source.astype(np.float32)).astype(source.dtype)
        graph.initializer.append(numpy_helper.from_array(value, node.output[0]))
        drop_ids.add(id(node))
        folded += 1
    if drop_ids:
        replace_nodes(graph, drop_ids)
    return folded


def prune_dead_nodes(graph):
    graph_outputs = {o.name for o in graph.output}
    while True:
        used = {i for node in graph.node for i in node.input if i}
        removable = [
            node
            for node in graph.node
            if node.output
            and all(
                output not in used and output not in graph_outputs
                for output in node.output
            )
        ]
        if not removable:
            return
        replace_nodes(graph, {id(node) for node in removable})


def prune_initializers_and_value_info(graph):
    graph_outputs = {o.name for o in graph.output}
    used = {i for node in graph.node for i in node.input if i}
    keep_names = used | {i.name for i in graph.input} | graph_outputs
    before = len(graph.initializer)
    kept_initializers = [t for t in graph.initializer if t.name in keep_names]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    live_values = keep_names | {o for node in graph.node for o in node.output}
    for vi in list(graph.value_info):
        if vi.name not in live_values:
            graph.value_info.remove(vi)
    return before - len(kept_initializers)


def fold_constant_trig_and_prune(graph):
    """Materialize constant Sin/Cos nodes and remove dead export leftovers.

    Torch exports rotary embedding caches as cached_freqs -> Slice -> Sin/Cos.
    ONNX Runtime's WebGPU session optimizer then tries to fold those constants
    through a CPU pass and warns when no CPU kernel is available for the fp16
    trig node. Folding them here keeps the published artifact quiet and avoids
    runtime startup work.
    """
    folded = fold_constant_trig_nodes(graph)
    prune_dead_nodes(graph)
    removed_initializers = prune_initializers_and_value_info(graph)
    return folded, removed_initializers


def main() -> None:
    args = parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.raw_output or output.with_name(f"{output.stem}_raw.onnx")

    print(
        f"=== export ONNX core at T={args.frames} (~{args.frames/100:.1f}s chunk) ==="
    )
    with open(args.config) as f:
        # FullLoader (not safe_load): config uses !!python/tuple; trusted local file.
        config = dict_to_namespace(yaml.load(f, Loader=yaml.FullLoader))  # noqa: S506
    model = MelBandRoformer(**vars(config.model)).eval()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    for m in model.modules():
        if isinstance(m, Attend):
            m.flash = False

    stft_bins = int(model.freq_indices.max().item()) + 1
    enc = torch.randn(1, stft_bins, args.frames, 2)
    with torch.no_grad():
        model.net_forward(enc)  # warm rotary cache at this seq_len

    class Core(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, stft_repr):
            return self.m.net_forward(stft_repr)

    t0 = time.time()
    torch.onnx.export(
        Core(model).eval(),
        (enc,),
        str(raw),
        input_names=["stft_repr"],
        output_names=["masks"],
        dynamo=True,
        opset_version=18,
    )
    print(f"exported fp32 in {time.time()-t0:.1f}s")
    del model, enc
    gc.collect()

    # late import: load conversion deps only after fp32 export
    import onnx  # noqa: PLC0415
    from onnxconverter_common import float16  # noqa: PLC0415

    block = list(
        dict.fromkeys(
            [*float16.DEFAULT_OP_BLOCK_LIST, "Softmax", "ReduceL2", "Clip", "Div"]
        )
    )
    m16 = float16.convert_float_to_float16(
        onnx.load(str(raw)),
        keep_io_types=True,
        disable_shape_infer=True,
        op_block_list=block,
    )
    nc, ns = split_wide_concat_split(m16.graph)
    nu = unwrap_softmax_fp16(m16.graph) if args.softmax == "fp16" else 0
    nt, ni = fold_constant_trig_and_prune(m16.graph)
    counts = Counter(n.op_type for n in m16.graph.node)
    print(
        f"patched: softmax={args.softmax}, concat-split trees(c={nc},s={ns}), "
        f"softmax-unwrap casts={nu}, folded trig={nt}, "
        f"pruned initializers={ni}; nodes={sum(counts.values())}"
    )

    def _save(m):
        output.unlink(missing_ok=True)
        Path(str(output) + ".data").unlink(missing_ok=True)
        onnx.save_model(
            m,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output.name + ".data",
            size_threshold=4096,
            convert_attribute=True,
        )

    _save(m16)
    # Post-save sanitize: the fp16 conversion materializes rare fp32 weight outliers
    # (>65504) as inf in the saved initializers -> ~0.45% NaN in masks (one band).
    # Reload the final model and zero any non-finite fp16 weights.
    reloaded = onnx.load(str(output), load_external_data=True)
    nfix = sanitize_fp16_initializers(reloaded)
    if nfix:
        _save(reloaded)
    print(f"sanitized {nfix} non-finite fp16 weight value(s) -> 0")
    raw.unlink(missing_ok=True)
    raw.with_suffix(".onnx.data").unlink(missing_ok=True)
    sz = (output.stat().st_size + Path(str(output) + ".data").stat().st_size) / 1e6
    print(f"wrote {output.name} (+data, {sz:.0f} MB)")
    print(
        "attention score buffer fp16 ~= "
        f"480*{args.frames}*{args.frames}*2 = "
        f"{480*args.frames*args.frames*2/1e6:.0f} MB"
    )


if __name__ == "__main__":
    main()
