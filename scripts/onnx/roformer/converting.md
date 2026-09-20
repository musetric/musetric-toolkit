# Converting a PyTorch Separator to ONNX

The toolkit keeps the torch implementation as the reference path and CLI
default. The files in this directory are offline tooling for exporting the
neural network core to ONNX and manually validating that core through Python
`onnxruntime`.

One graph is published, and the recipe below is the only way it is built: the
blocked, all-fp16 core at T = 1100 is what `musetric` loads on desktop and on a
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
| `syhft_core_t1100.onnx` | `88b51e87dd2fa02acecf95d880a3833c307bf780b101dd39021de7b821faec22` |
| `syhft_core_t1100.onnx.data` | `648db04fce69e556bc1fb08486ffd7f7ac50d370b1c6026e42ffea9cd621a7ed` |

Download both files into `tmp/models` (the `.data` file must sit next to its graph):

```bash
uv run hf download musetric/vocal-separation-roformer-onnx \
  syhft_core_t1100.onnx syhft_core_t1100.onnx.data \
  --local-dir tmp/models
```

T = 1100 is the published window.

## Install Tooling

The ONNX build dependencies live in the `export` dependency group, not in the
default install. `onnxruntime-gpu` (CUDA / DirectML / CPU execution providers)
ships in the default install, so syncing the group is enough to run or validate
the exported graph:

```bash
uv sync --group export
```

## Build a Core

Five steps: export, re-tree the wide `Concat`/`Split` nodes, audit the epsilon,
audit the dispatch rows, point the graph at the published weights file.

```bash
uv run --group export python scripts/onnx/roformer/build_full_onnx.py \
  --checkpoint tmp/models/MelBandRoformerBigSYHFTV1.ckpt \
  --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml \
  --output tmp/models/core_t1100.onnx \
  --core-only --fuse-rmsnorm --attn-block 64 --all-fp16 --frames 1100 --skip-gate \
  --split-rows 8 --split-projections 4

uv run --group export python scripts/onnx/roformer/split_concat_webgpu.py \
  --input tmp/models/core_t1100.onnx \
  --output tmp/models/syhft_core_t1100.onnx

uv run --group export python scripts/onnx/roformer/fp16_epsilon_audit.py \
  tmp/models/syhft_core_t1100.onnx

uv run --group export python scripts/onnx/roformer/dispatch_rows_audit.py \
  tmp/models/syhft_core_t1100.onnx

uv run python scripts/onnx/roformer/reuse_external_data.py \
  --model tmp/models/syhft_core_t1100.onnx \
  --reference published/syhft_core_t1100.onnx \
  --out tmp/models/release/syhft_core_t1100.onnx
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
  each at T = 1100.
- `--split-rows 8 --split-projections 4` shortens the longest dispatches; see
  below.
- `split_concat_webgpu.py` re-trees wide `Concat`/`Split` to <=8-wide so every
  shader stays at <=9 storage buffers, under the strictest shipping cap
  (Dawn/Metal on macOS reports `maxStorageBuffersPerShaderStage = 10`).
- `reuse_external_data.py` keeps the weights file of the published revision;
  see below.

## Keep Row-Dispatched Kernels Under 65535 Rows

The exporter applies this on every build; there is no flag. Several onnxruntime
WebGPU kernels run one workgroup per row and have no bounds check, because their
shared-memory reduction needs `workgroupBarrier()` in uniform control flow: the
normalizations, `Softmax`, `LpNormalization`, `TopK`, `InstanceNormalization`
and the `Reduce*` family. Past `maxComputeWorkgroupsPerDimension` rows (65535,
the WebGPU default and what Dawn/Metal reports) the EP turns the dispatch into a
`ceil(sqrt(rows))^2` square, and the spare workgroups write past the end of the
output. Dawn on Metal clamps that index to the last element:

- with the default bucketed storage cache the buffer is rounded up and the stray
  writes land in the slack, so nothing shows;
- with `storageBufferCacheMode: 'simple'` the buffer has the exact size and the
  last row is overwritten. `RMSNormalization` returns `+inf` there and attention
  spreads it as NaN over every mask; `Softmax` returns silently wrong values.

At T = 1100 the transformer norms see `[60, 1100, 384]` and `[1100, 60, 384]`,
66000 rows each, and the band-attention softmax sees `[1100, 8, 60, 60]`, 528000
rows. The exporter wraps every RMSNorm so it chunks its batch axis once the rows
pass the cap (two chunks here) and sets `Attend.softmax_rows`, which chunks only
the score tensor ahead of the softmax (nine chunks). Both are exact: on
Chrome/Metal the rebuilt core returns masks bit-identical to the previous one
with the default cache, now with `'simple'` too, and on D3D12 and on Adreno,
where the stray writes leave the output intact, the two cores agree bit for bit.
The weights file does not change. The cost is mostly memory: under `'simple'`
the buffer cache keeps a buffer of every new chunk size for the whole session,
about 270 MB of peak GPU memory, while wall clock moves by 2-5 %. Chunking q/k/v
instead of the scores measured 9 %.

`export()` refuses to write a graph where such a node sees more rows or has no
static input shape, and `dispatch_rows_audit.py` re-checks any artifact. The
`Reduce*` count is an upper bound: onnxruntime runs some reductions through a
naive kernel that has the guard.

## Shorten the Longest Dispatches

`--split-rows N` and `--split-projections N` are aimed at mobile GPUs, and the
published core is built with `--split-rows 8 --split-projections 4`. A run of
this core is
dominated by a handful of very long matmuls: the feed-forward projections and
the fused qkv projection of each layer are each a single dispatch tens of times
longer than the median one. Nothing at the runtime level can break those up -- a
flush threshold only decides how many dispatches ride in one submit, never how
long one of them runs -- so the compositor is locked out for as long as the
longest one takes, and that is what a freeze is.

The flag chunks every feed-forward over `N` row groups and replaces the fused
qkv projection with the three projections the following rearrange splits it into
anyway:

```bash
uv run --group export python scripts/onnx/roformer/build_full_onnx.py \
  --checkpoint tmp/models/MelBandRoformerBigSYHFTV1.ckpt \
  --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml \
  --output tmp/models/core_split4_t1100.onnx \
  --core-only --fuse-rmsnorm --attn-block 64 --all-fp16 --frames 1100 \
  --skip-gate --split-rows 4
```

`--split-rows` leaves the four projections of every attention -- q, k, v and
the output -- at full height: `[T * 60, 384] x [384, 512]` and its reverse, 66000
rows at T = 1100. Once the feed-forwards are chunked, those are the longest
dispatches of a run, about 200 ms each on Adreno 660. `--split-projections N`
chunks each of them into `N` row groups the same way. With
`--split-rows 8 --split-projections 4` the worst wait of another GPU client
during a paced run on Adreno 660 falls from about 650 ms to about 440 ms for
2-4 % of run time; finer splits cost more on the desktop without a steady gain
on the phones. The measurements are in musetric/musetric#894.

Both rewrites are exact in fp16, because rows of a matmul are independent and
the qkv split only skips a fusion. Splitting the *reduction* axis instead --
which is the obvious way to cut the same matmul -- is not exact in fp16: it
reorders the sums and costs roughly 66 dB a layer. Doing it on the exported
graph with `Split`/`Concat` nodes is exact but materializes the wide activation
twice, which costs several hundred MB of peak memory; doing it here costs none,
because the wide intermediate is never built at full height.

Run the same validation as any other core afterwards. Peak GPU memory falls
rather than rises, and total time moves by about a percent.

## Keep the Published Weights File

A re-export with the same weights lays them out in another order, and the
row-chunked projections keep their weight transposes as `Transpose` nodes that
the exporter otherwise folds into constants. The weights are the same, but the
`.onnx.data` is not, and an app that pins the new revision downloads 741 MB
again. `reuse_external_data.py` folds each such transpose into the matrix the
published core stores and points every tensor of the new graph at the offset of
the same bytes in the published weights file. It refuses when any tensor is not
found there. Only the `.onnx` then changes between revisions.

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
# --frames must match the model, and --out-dir is per-T. Large T (>=~901)
# needs --device cuda: the CPU torch forward segfaults on the 2.3 GB T²
# attention sim, while flash SDPA on cuda never materializes it.
uv run python scripts/onnx/roformer/validate_full_onnx.py \
  --checkpoint tmp/models/MelBandRoformerBigSYHFTV1.ckpt \
  --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml \
  --source tmp/sample.flac --out-dir tmp/bench_out_t1100 --frames 1100 --device cuda
```

It writes `full_input.f32` and `full_ref_vocals.f32` into `--out-dir`, planar
little-endian stereo, and prints the offset of the window it chose. Scoring an
execution provider against them takes a comparator that runs the ONNX and
measures SNR; this repository does not carry one.

## Inspect Ops

```bash
uv run --group export python scripts/onnx/roformer/op_audit.py tmp/models/syhft_core_t1100.onnx
```

## Run Python ONNX Inference

```bash
uv run --group export python scripts/onnx/roformer/infer_separator.py \
  --model tmp/models/syhft_core_t1100.onnx \
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
