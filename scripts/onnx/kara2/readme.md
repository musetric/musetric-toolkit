# UVR-MDX-NET KARA2 → WebGPU graph with time bands in the batch

The lead/backing separation step runs `UVR_MDXNET_KARA_2.onnx`, a convolutional
U-Net over a `[1, 4, 2048, 256]` complex spectrogram: 236 GMAC per chunk, most of
it in `3×3` convolutions over the full `256 × 2048` plane.

## Why the graph is rewritten

onnxruntime-web runs `Conv`, `ConvTranspose` and `BatchNormalization` on
WebGPU in NHWC and inserts a layout `Transpose` around each of them. For a
batch of one that `Transpose` reduces to a single `[C, H·W]` matrix swap, which
takes the tiled kernel that loses a quarter of its writes on Adreno 6xx. The
operators themselves compute correctly there. Probed one by one on an Adreno 660
with the model's own weights:

| Operator | Batch 1, NHWC | Batch 2, NHWC | Batch 1, NCHW layout |
|---|---|---|---|
| `Transpose` NCHW → NHWC | wrong, 5.9 dB | exact | — |
| `Conv` `3×3`, `2×2` stride 2, `1×1` | wrong, 5–8 dB | 143.5 dB to exact | 143.5 dB to exact, the `3×3` at half the speed |
| `ConvTranspose` `2×2` stride 2 | wrong, 4.1 dB | 137.2 dB | 137.2 dB |
| `BatchNormalization` | wrong, 5.5 dB | 145.1 dB | — |

The graph published up to revision `8f6d5fef` avoided those operators
altogether: a static rewrite (in this folder's history as
`rewrite_static_adreno.py`) unrolled every convolution into `MatMul`s. It was
correct everywhere and 1.1–2.1 times as slow as the original operators.

## The fold

`fold_time_bands.py` keeps every operator of the original graph and holds each
tensor of the U-Net body, `[1, C, T, F]` in the source, as `[B, C, T / B, F]`:
`B` consecutive bands of time frames, one per batch item, so every layout
transpose onnxruntime inserts transposes a batch of matrices.

| Source | In the fold |
|---|---|
| `Conv` `3×3`, pad 1 | halo rows from the neighbouring bands (zeros at the plane's ends) → `Concat` along time → the same `Conv` with no padding along time |
| `Conv` `2×2` stride 2, `ConvTranspose` `2×2` stride 2 | unchanged: they pair frames inside a band |
| frequency-axis `MatMul`, `BatchNormalization`, `Relu`, `Add`, `Mul` | unchanged: they act per frame |
| input `1×1` `Conv` and `Relu` | after the first `Transpose`, on the bands |
| output `1×1` `Conv` | a `MatMul` on the bands, then a rank-3 `Reshape` before the bands are joined back into time |

The output convolution is a `MatMul` because an NHWC convolution there lets
onnxruntime move its layout transpose past the join onto a batch of one, which
is wrong on Adreno 660 again. Every slice of the halo exchange keeps `B − 1`
bands, so `B` is at least 3; it must divide the frame count of every U-Net
level (256 down to 8).

```sh
uv run python scripts/onnx/kara2/fold_time_bands.py \
  --model UVR_MDXNET_KARA_2.onnx --out kara2.onnx --bands 4 --check
```

`--bands 4` is how the published graph is built: 428 nodes, the size of the
source. `--check` fails unless the CPU provider's output stays within `1e-4` of
the source's; with 4 bands it is exact.

On WebGPU the output equals the source graph's on the same device on desktop
NVIDIA, Adreno 750 and Apple M3, and is the same on Adreno 750 and Adreno 660.
Against the rewrite, three loads of each alternated:

| | RTX 3060 Laptop | Adreno 750 | Adreno 660 | Apple M3 (iPad Air) |
|---|---|---|---|---|
| Chunk, p50 | 670 → 329 ms | 3202 → 1926 ms | 8241 → 7211 ms | 3845 → 1815 ms |
| Foreign GPU job wait, p95 | 5 → 8 ms | 93 → 215 ms | 131 → 765 ms | 44 → 74 ms |
| Peak GPU memory | 1.69 → 1.21 GiB | 1.92 → 1.44 GiB | 1.90 → 1.42 GiB | — |

Each full-resolution convolution is one dispatch now, so a phone GPU stays
busy for longer stretches: the foreign job wait is the price of the speed.

## Publication

The graph is published as `kara2.onnx` in
[`musetric/uvr-mdxnet-kara2-onnx`](https://huggingface.co/musetric/uvr-mdxnet-kara2-onnx)
with the model card, built by `fold_time_bands.py --bands 4` from the UVR
release file `UVR_MDXNET_KARA_2.onnx` (sha256 `bf32e151…cbf5f4`): sha256
`f90e997a…e4ba12`, 52,832,612 bytes.
