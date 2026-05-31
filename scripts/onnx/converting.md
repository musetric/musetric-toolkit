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
