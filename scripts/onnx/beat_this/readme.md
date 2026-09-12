# Beat This! → ONNX export

Exports the **Beat This!** beat/downbeat tracker (the `beat-this` package the
`musetric-rhythm` CLI runs) to ONNX for the
[`musetric`](https://github.com/musetric/musetric) `@musetric/ai` runtime on
**onnxruntime-web**.

Like the chordmini export, the graph covers only the neural boundary; feature
extraction is the host's job:

```
beat_this.onnx  spect [windows, frames, 128] → beat, downbeat [windows, frames]
```

The runtime computes the log-mel on **WebGPU** (`@musetric/fft` STFT + the mel
filterbank exported here) and runs the graph on the WebGPU execution provider,
the same way chords runs its CQT on WebGPU around ChordNet. Chunking,
aggregation and peak picking also stay in the runtime: they are index
arithmetic, not DSP.

One window per session call is not a detail — it is what makes the tracker
runnable. Batching every window into a single call materializes an attention
tensor of `windows × 32 × 1500 × 1500` floats, which is ~2.9 GB on a
five-minute track.

## Why the filterbank ships as a file

`mel-filterbank.bin` is torchaudio's own `MelScale.fb`, written verbatim as
row-major float32 `[n_fft // 2 + 1, n_mels]`. Shipping it means the runtime
reuses the reference filterbank rather than reimplementing the slaney mel scale,
so the front end carries no approximation of its own. This mirrors how the
chordmini export ships `cqt-plan.bin`.

The analysis window is *not* shipped: the frame shader generates a periodic Hann
analytically, and the export fails if the reference window ever stops matching
that (`check_analytic_window`).

## Export-time swap

`PartialFTTransformer.forward` reads its batch size with `b = len(x)`, which
bypasses the tracer and freezes the window count into the graph. The export
swaps in the same forward written with `x.shape`, which traces to `Shape`/
`Gather`. The installed `beat_this` package is left untouched, and the patched
module is bit-identical to the vendored one in torch.

This one is worth remembering: `len()` does not raise, it *succeeds* and yields
a graph that returns wrong-length output for any track whose window count
differs from the traced one. Run the exported graph at several input lengths and
diff the shapes.

## Split does not compile on Adreno

onnxruntime's WebGPU `Split` kernel fails to build a compute pipeline on
Adreno. On a Galaxy S24 the graph loses its very first `Split` with
`CreateComputePipelines failed with VK_ERROR_UNKNOWN`, onnxruntime drops the
session to the CPU provider, and the tracker still returns beats — just far
slower, with nothing in the log naming the cause. This is a shader that will
not compile, not a binding too large, so nothing about tensor sizes helps.

`split_to_slice_webgpu.py` lowers every fixed-size `Split` into one static
`Slice` per output: it reads the sizes off the `Constant` feeding the node and
writes them as `starts`/`ends`/`axes` initializers. Same partition of the same
axis, no weight read, three initializers added per output. Run it on the export
before validating, so every later step sees the graph that ships.

## Adreno WebGPU rewrite

On Adreno 6xx the stock graph returns silently wrong logits: shared-tile
`Transpose`/`Conv`, inference `BatchNormalization`, and the 1500×1500
time-attention MatMul. The shipping rewrite pins **513 frames**, replaces the
broken Transpose/Conv families, and folds BatchNorm to Mul+Add. Rebuild:

```sh
uv run --group export python scripts/onnx/beat_this/capture_shapes.py \
  --model <export-dir>/beat_this.onnx --frames 513 --out shapes513.json
uv run --group export python scripts/onnx/beat_this/rewrite_static_adreno.py \
  --model <export-dir>/beat_this.onnx --shapes shapes513.json \
  --out <export-dir>/beat_this.onnx
```

The host must always feed `[1, 513, 128]` (pad short clips; long tracks use
overlapping 513-frame windows).

## Files

| script | role |
|---|---|
| `export_beat_this.py` | build the graph, write `beat_this.onnx`, `mel-filterbank.bin` + `config.json` |
| `split_to_slice_webgpu.py` | lower fixed-size `Split` into static `Slice` so the graph compiles on Adreno |
| `validate_beat_this.py` | beat/downbeat time agreement of the ONNX vs the torch `File2Beats` reference |
| `dump_reference_mel.py` | dump reference waveforms + log-mels, the parity gate for the runtime's WebGPU front end |
| `capture_shapes.py` | pin every intermediate shape at a chosen frame count |
| `rewrite_static_adreno.py` | replace broken Adreno Transpose/Conv/BatchNorm; ship at 513 frames |
| `stage_beat_this.py` | copy the file set into the publish folder |
| `publish_beat_this.py` | upload the staged folder to `musetric/beat-this-onnx` on HF |

All paths are explicit arguments; the scripts carry no implicit defaults.

## Run

The export reuses the main toolkit env (it imports `beat_this`) plus the
`export` dependency group (`onnx`):

```sh
uv run --group export python scripts/onnx/beat_this/export_beat_this.py \
  --out <export-dir> --models-path <checkpoint-cache-dir>

uv run --group export python scripts/onnx/beat_this/split_to_slice_webgpu.py \
  --input <export-dir>/beat_this.onnx --output <export-dir>/beat_this.onnx

uv run --group export python scripts/onnx/beat_this/validate_beat_this.py \
  --onnx <export-dir>/beat_this.onnx --audio <audio-dir> \
  --models-path <checkpoint-cache-dir>

uv run python scripts/onnx/beat_this/dump_reference_mel.py \
  --audio <audio-dir> --out <dump-dir> --models-path <checkpoint-cache-dir>

uv run python scripts/onnx/beat_this/stage_beat_this.py \
  --export <export-dir> --dest <publish-dir>

uv run python scripts/onnx/beat_this/publish_beat_this.py --src <publish-dir> --dry-run
```

Validate on the material the model is fed in production — the **instrumental
stem**.

`--models-path` sets `TORCH_HOME`, matching the CLI: the checkpoint
(`beat_this-final0.ckpt`, ~81 MB) is fetched there on first run. The exported
graph is ~83 MB fp32; the filterbank is 257 KB.

The reference downmixes stereo with `signal.mean(1)`, so the runtime must decode
with a mean downmix rather than `ffmpeg -ac 1`, which is √2 louder. `log1p`
features make that gain a non-constant offset; `config.json` records the
`downmix` contract.

Beat This! is used through the `beat-this` package; see `thirdPartyNotices.md`
for source and license.
