# Converting a PyTorch Separator to ONNX

The toolkit keeps the torch implementation as the reference path and CLI
default. The files in this directory are offline tooling for exporting the
neural network core to ONNX and manually validating that core through Python
`onnxruntime`.

One graph is published, and the recipe below is the only way it is built: the
blocked, all-fp16 core at T = 1101 is what `musetric` loads on desktop and on a
phone alike.

## Export Boundary

Only the pure neural network goes into the ONNX graph:

| method | role | runtime |
|---|---|---|
| `encode_stft(raw_audio) -> stft_repr` | host front-end | torch/Python |
| `net_forward(stft_repr) -> masks` | neural network core | ONNX |
| `decode_istft(stft_repr, masks) -> audio` | host back-end | torch/Python |

STFT, iSTFT, complex tensor handling, chunking, normalization, overlap-add, and
mask scatter stay outside the ONNX graph. The regular `forward` still calls the
same stages and remains the torch reference.

Attention uses the matmul path during export because it is portable across
onnxruntime execution providers. The normal torch path remains unchanged.

## Prebuilt Artifacts (published)

You do not have to rebuild anything. The core is published (MIT) at
<https://huggingface.co/musetric/vocal-separation-roformer-onnx>:

| File | SHA256 |
|---|---|
| `syhft_core_t1101.onnx` | `74305da0ca0d814eec99b6314ec41d8cbf8cc1a82dc60488e4f31b70439027a9` |
| `syhft_core_t1101.onnx.data` | `b08cfc80905e3560a4dd5d30f641299a47dd96d309ebbe9524d9d6c9d2a0356f` |

Download both files into `tmp/models` (the `.data` file must sit next to its graph):

```bash
uv run hf download musetric/vocal-separation-roformer-onnx \
  syhft_core_t1101.onnx syhft_core_t1101.onnx.data \
  --local-dir tmp/models
```

T = 1101 is the model's full reference context.

## Install Tooling

The ONNX build dependencies live in the `export` dependency group, not in the
default install. `onnxruntime-gpu` (CUDA / DirectML / CPU execution providers)
ships in the default install, so syncing the group is enough to run or validate
the exported graph:

```bash
uv sync --group export
```

## Build a Core

Three steps: export, re-tree the wide `Concat`/`Split` nodes, audit the epsilon.

```bash
uv run --group export python scripts/onnx/roformer/build_full_onnx.py \
  --checkpoint tmp/models/MelBandRoformerBigSYHFTV1.ckpt \
  --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml \
  --output tmp/models/core_t1101.onnx \
  --core-only --fuse-rmsnorm --attn-block 64 --all-fp16 --frames 1101 --skip-gate

uv run --group export python scripts/onnx/roformer/split_concat_webgpu.py \
  --input tmp/models/core_t1101.onnx \
  --output tmp/models/syhft_core_t1101.onnx

uv run --group export python scripts/onnx/roformer/fp16_epsilon_audit.py \
  tmp/models/syhft_core_t1101.onnx
```

What each flag is for:

- `--core-only` exports `stft_repr -> masks` rather than the full
  `raw_audio -> vocals` graph, because the host owns the STFT/iSTFT.
- `--attn-block 64` caps the attention score tensor at `[60, 8, 64, T]` instead
  of `[60, 8, T, T]`. Exact — softmax normalizes each query row over the full key
  axis on its own — and it is what keeps the graph inside a mobile
  storage-buffer binding, which stops working past 256 MiB whatever limit the
  adapter declares.
- `--fuse-rmsnorm` gives the normalization a single `ai.onnx::RMSNormalization`
  kernel that accumulates the sum of squares in `f32` inside the shader.
- `--all-fp16` drops the fp32 pins, which the fused RMSNorm makes safe. Without
  it the `[T, 60, 1536]` activations become 387 MiB fp32 tensors with a cast copy
  each at T = 1101.
- `split_concat_webgpu.py` re-trees wide `Concat`/`Split` to <=8-wide so every
  shader stays at <=9 storage buffers, under the strictest shipping cap
  (Dawn/Metal on macOS reports `maxStorageBuffersPerShaderStage = 10`).

**The epsilon is dtype-dependent and the audit is not optional.** In fp16 the
`RMSNormalization` reciprocal `1/sqrt(mean(x^2) + eps)` is cast back to fp16, so
any row under `1/65504^2 = 2.33e-10` becomes `+inf` and then `0 * inf = NaN`,
which attention spreads over the whole time axis — a silent output, not a
warning. `--all-fp16` therefore emits `RMSNORM_EPS_FP16` (1e-9) instead of
`RMSNORM_EPS` (1e-12); the exporter refuses to write a graph that breaks the
rule, and `fp16_epsilon_audit.py` re-checks any artifact, including ones built
elsewhere. This shipped broken once — `epsilon=1e-12` on all 96 fp16 nodes — and
`packages/mobile/docs/iosWebgpu.md` in the `musetric` repository carries the
device measurements.

## Validate

Gates: conversion SNR vs torch >= 40 dB, NaN = 0. The reference I/O comes from
Python:

```bash
# --frames must match the model; --out-dir is per-T (the parity test reads
# bench_out_t<frames>). Large T (>=~901) needs --device cuda: the CPU torch
# forward segfaults on the 2.3 GB T² attention sim, while flash SDPA on cuda
# never materializes it.
uv run python scripts/onnx/roformer/validate_full_onnx.py \
  --checkpoint tmp/models/MelBandRoformerBigSYHFTV1.ckpt \
  --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml \
  --source tmp/sample.flac --out-dir tmp/bench_out_t1101 --frames 1101 --device cuda
```

Then run `@musetric/ai`'s parity test (`yarn workspace @musetric/ai test`), which
loads `full_input.f32` / `full_ref_vocals.f32` and compares the WebGPU output.

## Inspect Ops

```bash
uv run --group export python scripts/onnx/roformer/op_audit.py tmp/models/syhft_core_t1101.onnx
```

## Run Python ONNX Inference

```bash
uv run --group export python scripts/onnx/roformer/infer_separator.py \
  --model tmp/models/syhft_core_t1101.onnx \
  --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml \
  --source path/to/input.wav \
  --target-output tmp/out/target.flac \
  --residual-output tmp/out/residual.flac
```

The script runs the exported ONNX core through Python `onnxruntime` with the
same host-side torch STFT/iSTFT stages used by the reference model. It is a
manual validation tool and is not wired into the default toolkit CLI.

## Notes

- Static shapes are intentional: a dynamic time axis is fragile for this
  architecture, because rotary and reshape logic bakes sequence lengths into the
  exported graph, so the window is fixed at build time.
- Repack external-data models before path-loading them with onnxruntime.
- Warm the model once at the export shape before `torch.onnx.export`.
- Delete stale `.onnx.data` files before saving external-data models; the scripts
  do this.
- Keep generated model files out of git unless explicitly publishing artifacts.
