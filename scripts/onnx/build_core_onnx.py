"""Build an ONNX core from a torch checkpoint and YAML config.

ckpt -> export NN core (dynamo, static prod-chunk shape) -> repack (small tensors
inline so onnxruntime loads by path / mmaps weights) -> op audit.

Boundary: only net_forward (the NN) goes to ONNX; STFT/iSTFT stay host-side
(torch here, another host runtime later). flash_attn=False -> matmul attention
(portable across onnxruntime EPs). See converting.md.

Run (from the repo root; needs an onnxruntime extra for the path-load check):
  uv run --group export --extra cpu python scripts/onnx/build_core_onnx.py \
    --checkpoint tmp/models/model.ckpt \
    --config tmp/models/config.yaml \
    --output tmp/models/model_core.onnx
"""

# ruff: noqa: T201 -- CLI build tool: stdout (progress/results) is its interface.

import argparse
import contextlib
import gc
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

from musetric_toolkit.separate_audio.roformer.attend import Attend
from musetric_toolkit.separate_audio.roformer.mel_band_roformer import MelBandRoformer
from musetric_toolkit.separate_audio.roformer_utils import dict_to_namespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a MelBand ONNX core.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Intermediate dynamo export path. Defaults next to --output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    raw = args.raw_output or output.with_name(f"{output.stem}_raw.onnx")
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        # FullLoader (not safe_load): config uses !!python/tuple; trusted local file.
        config = dict_to_namespace(yaml.load(f, Loader=yaml.FullLoader))  # noqa: S506

    model = MelBandRoformer(**vars(config.model)).eval()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    for m in model.modules():
        if isinstance(m, Attend):
            m.flash = False  # matmul attention path for portable export

    stft_bins = int(model.freq_indices.max().item()) + 1
    enc = torch.randn(
        1,
        stft_bins,
        config.inference.dim_t,
        2,
        dtype=torch.float32,
    )
    with torch.no_grad():
        model.net_forward(enc)  # warm RotaryEmbedding cache at this seq_len

    class Core(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, stft_repr):
            return self.m.net_forward(stft_repr)

    print(f"export core at stft_repr {tuple(enc.shape)} (dynamo, static) ...")
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
    print(f"  exported in {time.time()-t0:.1f}s")

    del model, enc
    gc.collect()

    # late import: load onnx only after freeing the torch model
    import onnx  # noqa: PLC0415

    print("repack (small tensors inline, weights external) ...")
    proto = onnx.load(str(raw))
    output.unlink(missing_ok=True)
    output.with_suffix(".onnx.data").unlink(missing_ok=True)
    onnx.save_model(
        proto,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output.name + ".data",
        size_threshold=4096,
        convert_attribute=True,
    )
    counts = Counter(n.op_type for n in proto.graph.node)
    del proto
    gc.collect()
    raw.unlink(missing_ok=True)
    raw.with_suffix(".onnx.data").unlink(missing_ok=True)

    total = output.stat().st_size + output.with_suffix(".onnx.data").stat().st_size
    print(f"  wrote {output.name} (+data, {total/1e6:.0f} MB)")
    print("op audit:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    if "Einsum" in counts:
        raise RuntimeError("Einsum present — matmul rewrite did not take")
    print("  no Einsum (matmul attention) — portable across EPs")

    # late import: load onnxruntime only after freeing the torch model
    import onnxruntime as ort  # noqa: PLC0415

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        str(output), sess_options=so, providers=["CPUExecutionProvider"]
    )
    print("ORT path-load OK:", [(i.name, i.shape) for i in sess.get_inputs()])


if __name__ == "__main__":
    main()
