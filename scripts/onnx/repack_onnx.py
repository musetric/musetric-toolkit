"""Repack a dynamo-exported ONNX so onnxruntime can load it BY PATH.

dynamo externalizes ALL tensors (even tiny shape/Split constants like val_21),
so ORT's load-time shape inference fails ("Cannot parse data from external
tensors"), forcing a SerializeToString() bytes load that needs ~1.5GB extra RAM
(MemoryError on this box). Re-saving with size_threshold keeps small tensors
inline (big weights stay external) -> ORT loads by path, mmapping weights.

build_core_onnx.py already does this inline; this is the standalone step.

Run (no torch needed; needs an onnxruntime extra for the path-load check):
  uv run --group export --extra cpu python scripts/onnx/repack_onnx.py \
    --input tmp/models/model_raw.onnx \
    --output tmp/models/model_core.onnx
"""

# ruff: noqa: T201 -- CLI build tool: stdout (progress/results) is its interface.

import argparse
from pathlib import Path

import onnx
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repack an ONNX external-data model.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


args = parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

print(f"loading {args.input.name} (+external data) ...")
proto = onnx.load(str(args.input))

print("re-saving (small tensors inline, weights external) ...")
args.output.unlink(missing_ok=True)
args.output.with_suffix(".onnx.data").unlink(missing_ok=True)
onnx.save_model(
    proto,
    str(args.output),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=args.output.name + ".data",
    size_threshold=4096,  # tensors <4KB stay inline (shape consts, split sizes)
    convert_attribute=True,
)
sz = args.output.stat().st_size + (args.output.with_suffix(".onnx.data").stat().st_size)
print(f"  wrote {args.output.name} (+data, {sz/1e6:.0f} MB)")

print("verifying ORT path-load ...")
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(
    str(args.output), sess_options=so, providers=["CPUExecutionProvider"]
)
print("OK path-load. inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
print("       outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
