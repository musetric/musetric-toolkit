# UVR-MDX-NET KARA2 → WebGPU-safe static graph

The lead/backing separation step runs `UVR_MDXNET_KARA_2.onnx`, a convolutional
U-Net over a `[1, 4, 2048, 256]` complex spectrogram. On Adreno 6xx the
onnxruntime-web WebGPU kernels for `BatchNormalization`, `Conv` and
`ConvTranspose` corrupt this graph nondeterministically, while `Transpose`,
`MatMul`, `Relu`, `Add` and `Mul` compute it correctly.

`rewrite_static_adreno.py` rebuilds the graph from the working operators only:

| Source | Rewrite |
|---|---|
| `BatchNormalization` | `Mul` + `Add` with folded running statistics |
| `Conv` | sum over kernel offsets: `Pad` → `Slice` with the stride → `Reshape` to `[C_in, H·W]` → weight-left `MatMul` |
| `ConvTranspose`, kernel = stride | one weight-left `MatMul` per kernel offset, interleaved with `Concat` + `Reshape` |

The rewrite pins batch 1 and the source input shape. It adds no transposition
and no per-position index tables, so the file size stays that of the source.

```sh
uv run python scripts/onnx/kara2/rewrite_static_adreno.py \
  --model UVR_MDXNET_KARA_2.onnx --out kara2_adreno.onnx --check
```

`--check` runs the source and the rewrite on the CPU provider with the same
deterministic input and fails when they differ by more than `1e-3`. Device
parity is checked separately: the rewrite on WebGPU against the wasm provider
on the same device, with the same input bytes.

## Publication

The rewrite is published as
[`musetric/uvr-mdxnet-kara2-onnx`](https://huggingface.co/musetric/uvr-mdxnet-kara2-onnx)
with the model card. The graph there is built by this script from the UVR
release file
`UVR_MDXNET_KARA_2.onnx` (sha256 `bf32e151…cbf5f4`) and saved as `kara2.onnx`.
