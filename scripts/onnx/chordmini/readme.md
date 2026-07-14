# ChordMini → ONNX/CQT export

This directory exports the vendored **ChordMini** chord recognizer (ChordNet
"2E1D", 170-class large vocabulary) as a **ChordNet/CQT artifact**: a small
`features → logits` ONNX classifier plus the versioned CQT plan that defines its
input features. Feature extraction runs on WebGPU in `@musetric/cqt`, not in the
graph.

All source and destination paths are explicit arguments. The scripts do not
guess folders or reuse a machine-specific cache, so a model cannot silently be
combined with a stale CQT plan.

## The artifact

`export_chordnet.py` produces the classifier boundary used after the WebGPU CQT
pipeline:

```text
features [W, 108, 144]                  # unnormalized log-CQT windows
  → (x − mean) / (std + 1e-8)           # normalization is inside ONNX
  → ChordNet
  → logits [W, 108, 170]
```

`W` is dynamic, while the 108-frame window and 144 CQT bins are fixed model
contract dimensions. CQT extraction, tail padding/windowing, temporal smoothing,
and argmax intentionally remain outside ONNX so the WebGPU runtime can keep the
feature and logits buffers on the GPU. `config.json` records the input/output
names and shapes, checkpoint normalization, and the 170-label vocabulary.

The staged and published file set is:

| file | role |
|---|---|
| `chordnet.onnx` | `features [W, 108, 144] float32 → logits [W, 108, 170] float32` |
| `config.json` | ChordNet input/output and normalization contract |
| `cqt-plan.bin` | little-endian sparse CQT-plan binary |
| `cqt-plan.manifest.json` | plan format, generator provenance, and hashes |

The binary plan and its manifest are published with the classifier so each
release has a hashable model/feature-extraction contract. The generated
TypeScript plan payload is for the `@musetric/cqt` tests; it is intentionally not
copied into the HF artifact.

## Scripts

| script | role |
|---|---|
| `export_chordnet.py` | write `chordnet.onnx` + `config.json` |
| `validate_chordnet.py` | compare deterministic feature-window logits from the exported classifier against the same Torch ChordNet pipeline |
| `generate_cqt_plan.py` | generate the binary CQT plan, its JSON provenance manifest, and the TypeScript payload the `@musetric/cqt` tests use |
| `generate_cqt_reference.py` | generate the librosa peak-magnitude table the `@musetric/cqt` tests measure against |
| `stage_chordnet.py` | stage the ChordNet/CQT bundle under its published names |
| `publish_chordnet.py` | validate and upload the staged bundle |

## Build, stage, and publish

The ONNX export uses the main toolkit environment plus the `export` dependency
group (`onnx`). The plan generator uses the pinned toolkit `librosa`/`numpy`/
`scipy` environment.

```sh
uv run --group export python scripts/onnx/chordmini/export_chordnet.py \
  --out <chordnet-export-dir> --models-path <checkpoint-cache-dir>

uv run --group export python scripts/onnx/chordmini/validate_chordnet.py \
  --onnx <chordnet-export-dir>/chordnet.onnx \
  --models-path <checkpoint-cache-dir>

uv run python scripts/onnx/chordmini/generate_cqt_plan.py \
  --out <cqt-plan.bin> --manifest <cqt-plan.manifest.json> \
  --typescript-payload <packages/cqt/src/cqt/__test__/planPayload.ts>

uv run python scripts/onnx/chordmini/generate_cqt_reference.py \
  --out <packages/cqt/src/cqt/__test__/reference.ts>

uv run python scripts/onnx/chordmini/stage_chordnet.py \
  --chordnet-export <chordnet-export-dir> \
  --cqt-plan <cqt-plan.bin> \
  --cqt-plan-manifest <cqt-plan.manifest.json> \
  --dest <publish-dir>

uv run python scripts/onnx/chordmini/publish_chordnet.py \
  --src <publish-dir> --dry-run
```

`stage_chordnet.py` always renames the two explicit plan inputs to the stable
published names `cqt-plan.bin` and `cqt-plan.manifest.json`. `publish_chordnet.py`
targets `musetric/chordmini-onnx` by default; pass `--repo <owner/repo>` to select
another model repo.

`validate_chordnet.py` runs three fixed-seed random unnormalized feature windows
through the checkpoint-backed Torch classifier and CPU ONNX Runtime. It checks
the `[3, 108, 170]` contract and fails when the maximum absolute logit error
exceeds `1e-4`.

## Validating end to end

The classifier check above covers the export, not the features. End-to-end
parity against the torch + librosa runner is measured from the TypeScript side by
`validateChordNetWebGpu` in `@musetric/ai`, which runs the real WebGPU CQT.

Validate on the material fed in production — the instrumental stem. Agreement
measured on audio where the reference emits a near-constant label (for example,
an isolated vocal where "no chord" is correct) carries no useful information: a
stub returning that label scores just as well.

The checkpoint (`2e1d_model_best.pth`, ~28 MB) is fetched on first run by
`ensure_checkpoint`. ChordMini is vendored under
`musetric_toolkit/chords_audio/chordmini`; see `thirdPartyNotices.md` for source
and license.
