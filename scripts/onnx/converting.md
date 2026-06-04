# Converting a PyTorch Separator to ONNX

The toolkit keeps the torch implementation as the reference path and CLI
default. The files in this directory are offline tooling for exporting a neural
network core to ONNX and manually validating that core through Python
`onnxruntime`.

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

## Prebuilt Artifact (published)

You do not have to rebuild the core. The first separation artifact is published
(MIT) at:

- Repo: <https://huggingface.co/musetric/vocal-separation-roformer-onnx>
- `syhft_core_fp16_t501.onnx` — SHA256 `4e6d3df35bca530893ea2a55bd1d7a78bc3721efbd51c8d3ed10eb3a19fa6d79`
- `syhft_core_fp16_t501.onnx.data` — SHA256 `1bbc7fed448872976b28710d03d9ec8b41f513dab9a3a9f0ff6493c8b5e5e22d`

Download both into `tmp/models` (the `.data` file must sit next to the graph):

```bash
uv run hf download musetric/vocal-separation-roformer-onnx \
  syhft_core_fp16_t501.onnx syhft_core_fp16_t501.onnx.data \
  --local-dir tmp/models
```

Then run Python ONNX inference on it (config comes from the source checkpoint
download, see `download_big_syhft`):

```bash
uv run --group export --extra cpu python scripts/onnx/infer_separator.py \
  --model tmp/models/syhft_core_fp16_t501.onnx \
  --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml \
  --source path/to/input.flac \
  --target-output tmp/out/vocals.flac \
  --residual-output tmp/out/instrumental.flac
```

## Install Tooling

The ONNX build dependencies live in the `export` dependency group, not in the
default install. Install one runtime extra only when you want to run or validate
the exported graph:

```bash
uv sync --group export --extra cpu
uv sync --group export --extra cuda
```

## Build

```bash
uv run --group export --extra cpu python scripts/onnx/build_core_onnx.py \
  --checkpoint tmp/models/model.ckpt \
  --config tmp/models/config.yaml \
  --output tmp/models/model_core.onnx
```

This exports `net_forward` with a static shape, repacks small tensors inline so
onnxruntime can path-load the graph, and checks that the graph contains no
`Einsum` nodes.

## Convert to FP16

```bash
uv run --group export --extra cpu python scripts/onnx/convert_fp16.py \
  --input tmp/models/model_core.onnx \
  --output tmp/models/model_core_fp16.onnx
```

The converter keeps graph inputs and outputs as FP32 while converting internal
weights and activations. It also sanitizes non-finite FP16 initializers after
conversion.

## Patch Wide Concat/Split Nodes

```bash
uv run --group export python scripts/onnx/split_concat_webgpu.py \
  --input tmp/models/model_core_fp16.onnx \
  --output tmp/models/model_core_fp16_webgpu.onnx
```

This rewrites wide `Concat` and `Split` nodes into small trees. The rewrite is
associative and preserves values while avoiding execution-provider buffer-count
limits.

## Export a Custom Static Chunk

```bash
uv run --group export python scripts/onnx/export_chunk.py \
  --checkpoint tmp/models/model.ckpt \
  --config tmp/models/config.yaml \
  --output tmp/models/model_core_fp16_t501.onnx \
  --frames 501 \
  --softmax fp16
```

Use this when a different fixed sequence length is needed. Static shapes are
intentional: dynamic time axes are fragile for this architecture because rotary
and reshape logic can bake sequence lengths into the exported graph.

## Full-Graph Build (single ONNX, all on WebGPU)

`@musetric/ai` runs the whole separation in **one** ONNX graph on the
onnxruntime-node WebGPU EP (audio in -> vocals out, 1 CPU crossing per chunk).
STFT and iSTFT are folded into the graph as `Conv1d` / `ConvTranspose1d` with
precomputed window x DFT-basis weights (the native `STFT`/`DFT` ops fall back to
CPU on the WebGPU EP; Conv/ConvTranspose/Scatter/Gather do not). Mask-apply uses
real-arithmetic `scatter_add`. The NN core (`net_forward`) is reused unchanged.

```bash
# 1. numerics gate (torch FullSeparator vs model.forward, CPU) — should print PASS
uv run python scripts/onnx/build_full_onnx.py --check \
  --checkpoint tmp/models/MelBandRoformerBigSYHFTV1.ckpt \
  --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml

# 2. export + fp16 + fusion (production). --fuse rewrites, before fp16 conversion:
#      erf-gelu       -> com.microsoft FastGelu
#      RMSNorm chain  -> ONNX RMSNormalization (opset 23)
#      attention core -> com.microsoft MultiHeadAttention (flash, no T² sim)
#    all single WebGPU kernels. Fusion lowers Cast 626->14, nodes 3158->~2018,
#    SNR ~52 dB vs torch after the RMSNorm-eps fix (was ~45 before, ~51 unfused;
#    well past the 40 dB gate).
#    The flash MHA removes the VRAM paging cliff, so larger --frames fit on
#    modest GPUs: T=1101 is the reference 11 s context. Build one model per T you
#    want to ship (the runtime picks by SYHFT_FRAMES). DEFAULT IS 1101 (chosen for
#    quality: smaller T loses separation context — full-track vs torch fp32 T=1101
#    is ~28 dB @ T=901, ~24 dB @ T=501; fp16 itself is ~48 dB at any T). Drop --fuse
#    for the old unfused build.
uv run --group export --extra cpu python scripts/onnx/build_full_onnx.py \
  --checkpoint tmp/models/MelBandRoformerBigSYHFTV1.ckpt \
  --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml \
  --output tmp/models/syhft_full_fp16_t1101.onnx --fuse --frames 1101  # default (quality)

# 3. WebGPU buffer-cap patch -> the model @musetric/ai loads
uv run --group export python scripts/onnx/split_concat_webgpu.py \
  --input tmp/models/syhft_full_fp16_t1101.onnx \
  --output tmp/models/syhft_full_webgpu_t1101.onnx
```

Note: `--fuse` bumps the onnx opset to 23 (RMSNormalization), which makes the
rotary `Cos`/`Sin` and the iSTFT `ConvTranspose` (11 nodes, < 2 %) fall back to
CPU on the WebGPU EP. Net is still faster; a future pass can constant-fold the
rotary trig and find an opset-18 iSTFT to keep everything on WebGPU. Verify EP
placement + SNR with `packages/ai/src/bench/profileProviders.ts`.

### Web-backend core (`onnxruntime-web` + `@musetric/fft`)

The `web` backend (`SYHFT_BACKEND=web`) runs only `net_forward` in ONNX and does
STFT/iSTFT host-side in `@musetric/fft`, so it needs the **core** graph, not the
full one. Build it with `--core-only` (same `--fuse` fp16/fusion pipeline, but
exports `stft_repr -> masks` instead of `raw_audio -> vocals`):

```bash
uv run --group export --extra cpu python scripts/onnx/build_full_onnx.py \
  --checkpoint tmp/models/MelBandRoformerBigSYHFTV1.ckpt \
  --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml \
  --output tmp/models/syhft_core_fused_fp16.onnx --fuse --core-only --frames 1101 --skip-gate
uv run --group export python scripts/onnx/split_concat_webgpu.py \
  --input tmp/models/syhft_core_fused_fp16.onnx \
  --output tmp/models/syhft_core_fused_fp16_webgpu.onnx
```

Validate on the WebGPU EP (gates: conversion SNR vs torch >= 40 dB, NaN=0, every
node on `WebGpuExecutionProvider`). The reference I/O comes from Python:

```bash
# --frames must match the model; --out-dir is per-T (the parity test reads
# bench_out_t<frames>). Large T (>=~901) needs --device cuda: the CPU torch
# forward segfaults on the 2.3 GB T² attention sim, while flash SDPA on cuda
# never materializes it.
uv run python scripts/onnx/validate_full_onnx.py \
  --checkpoint tmp/models/MelBandRoformerBigSYHFTV1.ckpt \
  --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml \
  --source tmp/sample.flac --out-dir tmp/bench_out_t501 --frames 501
# quality build: --out-dir tmp/bench_out_t1101 --frames 1101 --device cuda
```

then run `@musetric/ai`'s parity test (`yarn workspace @musetric/ai test`),
which loads `full_input.f32` / `full_ref_vocals.f32` and compares the WebGPU
output. The `net_forward`-only artifact above remains the published core; the
full graph is the Node runtime path.

## Inspect Ops

```bash
uv run --group export python scripts/onnx/op_audit.py tmp/models/model_core.onnx
```

## Run Python ONNX Inference

```bash
uv run --group export --extra cpu python scripts/onnx/infer_separator.py \
  --model tmp/models/model_core.onnx \
  --config tmp/models/config.yaml \
  --source path/to/input.wav \
  --target-output tmp/out/target.flac \
  --residual-output tmp/out/residual.flac
```

The script runs the exported ONNX core through Python `onnxruntime` with the
same host-side torch STFT/iSTFT stages used by the reference model. It is a
manual validation tool and is not wired into the default toolkit CLI.

## Notes

- Repack external-data models before path-loading them with onnxruntime.
- Warm the model once at the export shape before `torch.onnx.export`.
- Delete stale `.onnx.data` files before saving external-data models; the scripts
  do this.
- Keep generated model files out of git unless explicitly publishing artifacts.
