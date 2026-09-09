"""Build the FULL separation ONNX graph: raw stereo audio -> vocals, with STFT,
the NN core, mask-apply and iSTFT all inside one graph so onnxruntime-node runs
it end-to-end on the WebGPU EP (1 CPU crossing/chunk, no second device).

STFT/iSTFT are implemented as Conv1d / ConvTranspose1d with precomputed
window x DFT-basis weights (NOT torch.stft/istft, which export to the WebGPU-
unsupported STFT/DFT ops). Mask-apply uses real-arithmetic scatter_add (complex
ops don't export to ONNX). The NN core (`net_forward`) is reused unchanged.
"""

# ruff: noqa: T201

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from einops import rearrange
from torch import nn

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

from musetric_toolkit.separate_audio.roformer.attend import Attend
from musetric_toolkit.separate_audio.roformer.mel_band_roformer import (
    Attention,
    FeedForward,
    MelBandRoformer,
)
from musetric_toolkit.separate_audio.roformer_utils import dict_to_namespace

N_FFT = 2048
HOP = 441
WIN = 2048
# Attention head config (from config_vocals_mel_band_roformer_big_v1_ft.yaml:
# heads=8, dim_head=64 -> dim_inner=512), used by the MHA fusion.
HEADS = 8
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM  # 512
T = 1101  # default frame count, the published window; override with --frames
TSAMP = HOP * (T - 1)  # 220500
PAD = N_FFT // 2  # 1024
FREQS = N_FFT // 2 + 1  # 1025
PACKED = FREQS * 2  # 2050
# RMSNorm epsilon. The model's RMSNorm is F.normalize(x, dim=-1) =
# x/max(||x||_2, 1e-12), so the ONNX form x/sqrt(mean(x^2)+eps) wants the smallest
# epsilon it can carry: at 1e-6 every row whose per-element rms is below ~0.03 is
# corrupted and those rows exist (band-split / early features), which measured
# ~10 dB end-to-end.
RMSNORM_EPS = 1e-12
# ... but 1e-12 is only usable while the node computes in fp32. A WebGPU kernel
# normalizes by the reciprocal 1/sqrt(mean(x^2)+eps) and casts it to the tensor
# dtype, so with fp16 tensors any row whose mean(x^2)+eps falls below
# 1/65504^2 = 2.33e-10 turns that reciprocal into +inf, then 0*inf = NaN for the
# whole row, and the attention softmax spreads the NaN over the entire time axis.
# The bound does not depend on the data: 1/sqrt(eps) alone has to stay inside
# fp16. Measured with a one-node fp16 graph on a WebGPU adapter: whole-row NaN at
# eps <= 2.0e-10, clean from 2.4e-10 up. 1e-9 leaves the reciprocal a 2x margin
# and costs nothing measurable - core parity against torch fp32 on a real chunk is
# 45.68 dB at 1e-9 against 45.70 dB at 1e-12, while 1e-8 drops to 43.88 dB and
# 1e-7 to 38.86 dB, past the 40 dB gate. Anything near the 1e-4 a device-side
# patch first used is far outside it.
RMSNORM_EPS_FP16 = 1e-9
FP16_RSQRT_FLOOR = 1.0 / 65504.0**2  # 2.33e-10


def hann_periodic() -> np.ndarray:
    return np.hanning(WIN + 1)[:-1].astype(np.float64)


def stft_conv_weight() -> torch.Tensor:
    # [2050,1,2048]: out bins 0..1024 = Re, 1025..2049 = Im (per channel).
    window = hann_periodic()
    n = np.arange(N_FFT)
    f = np.arange(FREQS)
    angle = 2 * np.pi * np.outer(f, n) / N_FFT
    w_real = np.cos(angle) * window
    w_imag = -np.sin(angle) * window
    w = np.concatenate([w_real, w_imag], axis=0)[:, None, :]
    return torch.from_numpy(w.astype(np.float32))


def istft_conv_weight() -> torch.Tensor:
    # [2050,1,2048]: in bins 0..1024 = Re, 1025..2049 = Im; folds synthesis window.
    window = hann_periodic()
    n = np.arange(N_FFT)
    k = np.arange(FREQS)
    a = np.full(k.shape, 2.0)
    a[0] = 1.0
    a[-1] = 1.0
    angle = 2 * np.pi * np.outer(k, n) / N_FFT
    coeff_real = (a[:, None] / N_FFT) * np.cos(angle)
    coeff_imag = -(a[:, None] / N_FFT) * np.sin(angle)
    w = np.concatenate([coeff_real, coeff_imag], axis=0) * window[None, :]
    return torch.from_numpy(w[:, None, :].astype(np.float32))


def window_envelope() -> torch.Tensor:
    window = hann_periodic()
    ones = torch.ones(1, 1, T, dtype=torch.float64)
    wsq = torch.from_numpy(window**2).view(1, 1, N_FFT)
    env = F.conv_transpose1d(ones, wsq, stride=HOP)[0, 0, PAD : PAD + TSAMP]
    return env.clamp(min=1e-8).float()


class FullSeparator(nn.Module):
    """raw_audio [1,2,TSAMP] -> vocals [1,2,TSAMP], all real ops."""

    def __init__(self, model: MelBandRoformer):
        super().__init__()
        self.model = model
        self.register_buffer("stft_w", stft_conv_weight())
        self.register_buffer("istft_w", istft_conv_weight())
        self.register_buffer("env", window_envelope())
        # denom[2050] = num_bands_per_freq repeated per channel (s inner).
        nbpf = model.num_bands_per_freq.float()  # [1025]
        denom = nbpf.repeat_interleave(2).clamp(min=1e-8)  # [2050]
        self.register_buffer("denom", denom)
        self.register_buffer("freq_indices", model.freq_indices.long())  # [3958]

    def stft(self, raw_audio: torch.Tensor) -> torch.Tensor:
        # [1,2,TSAMP] -> stft_repr [1,2050,T,2] packed (f*2+s, c=re/im).
        x = raw_audio.reshape(2, 1, TSAMP)
        x = F.pad(x, (PAD, PAD), mode="reflect")
        y = F.conv1d(x, self.stft_w, stride=HOP)  # [2,2050,T]  (re0..1024,im0..1024)
        y = y.reshape(1, 2, 2, FREQS, T)  # b, ch, reim, f, T
        y = y.permute(0, 3, 1, 4, 2)  # b, f, ch, T, reim
        return y.reshape(1, PACKED, T, 2)

    def apply_mask_istft(
        self, stft_repr: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        # stft_repr [1,2050,T,2]; masks [1,1,3958,T,2] -> vocals [1,2,TSAMP].
        masks = masks.reshape(1, self.freq_indices.shape[0], T, 2)  # [1,3958,T,2]
        index = self.freq_indices.view(1, -1, 1, 1).expand_as(masks)
        summed = torch.zeros(
            1, PACKED, T, 2, dtype=masks.dtype, device=masks.device
        ).scatter_add(
            1, index, masks
        )  # [1,2050,T,2]
        averaged = summed / self.denom.view(1, PACKED, 1, 1)
        sr, si = stft_repr[..., 0], stft_repr[..., 1]
        mr, mi = averaged[..., 0], averaged[..., 1]
        out_r = sr * mr - si * mi
        out_i = sr * mi + si * mr  # [1,2050,T]
        # [1,2050,T] (f*2+s) -> [2(s),2050(re/im),T] for conv_transpose
        rr = out_r.reshape(1, FREQS, 2, T)
        ii = out_i.reshape(1, FREQS, 2, T)
        re = rr.permute(0, 2, 1, 3).reshape(2, FREQS, T)  # [s, f, T]
        im = ii.permute(0, 2, 1, 3).reshape(2, FREQS, T)
        stacked = torch.cat([re, im], dim=1)  # [2, 2050, T]
        # iSTFT as ConvTranspose1d (iDFT + synthesis-window + overlap-add fused).
        # NOTE: this op lands on the CPU EP (the opset-23 bump drops it there) at
        # ~88 ms/chunk — and that is the better placement. Tried replacing it with a
        # MatMul iDFT (~25 ms on WebGPU, 8x faster than the WebGPU ConvTranspose's
        # 200 ms) + a manual pad/reshape/shifted-add overlap-add: end-to-end it was a
        # WASH (the fold's memory movement ~= the CPU iSTFT it removes, on this
        # movement-bound graph) AND cost ~5 dB (fp16 matmul). The CPU ConvTranspose
        # does iDFT+OLA in one efficient MKL call; the overlap-add — not the DFT —
        # is the real cost. Don't re-try without a cheaper-than-movement OLA.
        y = F.conv_transpose1d(stacked, self.istft_w, stride=HOP)  # [2,1,..]
        y = y[:, 0, PAD : PAD + TSAMP] / self.env  # [2, TSAMP]
        return y.reshape(1, 2, TSAMP)

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        stft_repr = self.stft(raw_audio)
        masks = self.model.net_forward(stft_repr)
        return self.apply_mask_istft(stft_repr, masks)


def set_attention_block(model: MelBandRoformer, q_block: int) -> int:
    """Split the query axis of every attention layer into q_block-row chunks.

    Exact, not an approximation — see Attend.forward. The point is the peak score
    tensor: this model's six time-attention layers otherwise allocate
    [60, 8, T, T] fp16, which is 230 MiB at T=501 and 1110 MiB at T=1101, past
    what a mobile WebGPU storage buffer will bind. The six band-attention layers
    are [T, 8, 60, 60] and were never the problem, so blocking skips them (their
    sequence is 60, below any sensible block).
    """
    count = 0
    for m in model.modules():
        if isinstance(m, Attend):
            m.q_block = q_block
            count += 1
    return count


class RowChunkedFeedForward(nn.Module):
    """Run a feed-forward block over row chunks instead of the whole activation.

    Exact, and exact in fp16 too: rows of a matmul are independent, so every
    output row is produced by the same arithmetic as before. Splitting the
    reduction axis instead reorders the sums and costs roughly 66 dB a layer.

    The point is dispatch length. One feed-forward is a [rows, 384] x [384, 1536]
    matmul, which a mobile GPU runs as a single uninterruptible half-second of
    work; a queue-level flush threshold cannot reach inside it. Chunking also
    keeps the wide [rows, 1536] intermediate from being materialized at full
    height, so peak memory falls instead of rising.
    """

    def __init__(self, net: nn.Sequential, parts: int) -> None:
        super().__init__()
        self.net = net
        self.parts = parts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.net(c) for c in x.chunk(self.parts, dim=0)], dim=0)


def _attention_forward_unfused(self: Attention, x: torch.Tensor) -> torch.Tensor:
    """Attention.forward with the fused qkv projection replaced by three."""
    x = self.norm(x)
    q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
    k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=self.heads)
    v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=self.heads)
    if self.rotary_embed is not None:
        q = self.rotary_embed.rotate_queries_or_keys(q)
        k = self.rotary_embed.rotate_queries_or_keys(k)
    out = self.attend(q, k, v)
    gates = self.to_gates(x)
    out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()
    out = rearrange(out, "b h n d -> b n (h d)")
    return self.to_out(out)


def split_heavy_projections(model: MelBandRoformer, parts: int) -> tuple[int, int]:
    """Cut the two longest matmuls of every transformer layer into shorter ones.

    Feed-forwards are chunked by rows into `parts`. The fused qkv projection is
    replaced by the three projections the following rearrange splits it into
    anyway, so nothing downstream wants the fused tensor back.

    Both rewrites are exact and neither adds a Concat over a wide activation.
    Doing the same on the exported graph with Split/Concat instead does, and
    pays several hundred MB of peak memory for it.
    """
    feeds = 0
    for module in list(model.modules()):
        if isinstance(module, FeedForward):
            module.net = RowChunkedFeedForward(module.net, parts)
            feeds += 1
    attentions = 0
    for module in list(model.modules()):
        if not isinstance(module, Attention):
            continue
        weight = module.to_qkv.weight.data
        inner = weight.shape[0] // 3
        for name, index in (("to_q", 0), ("to_k", 1), ("to_v", 2)):
            linear = nn.Linear(module.to_qkv.in_features, inner, bias=False)
            linear.weight.data.copy_(weight[index * inner : (index + 1) * inner])
            setattr(module, name, linear)
        del module.to_qkv
        module.forward = _attention_forward_unfused.__get__(module, Attention)
        attentions += 1
    return feeds, attentions


def load_model(checkpoint: Path, config: Path) -> MelBandRoformer:
    with open(config) as f:
        cfg = dict_to_namespace(yaml.load(f, Loader=yaml.FullLoader))  # noqa: S506
    model = MelBandRoformer(**vars(cfg.model)).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    for m in model.modules():
        if isinstance(m, Attend):
            m.flash = False
    return model


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    err = ((ref - got) ** 2).sum().clamp(min=1e-20)
    return (10 * torch.log10((ref**2).sum() / err)).item()


def gate1(model: MelBandRoformer, full: FullSeparator) -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 2, TSAMP) * 0.1
    with torch.no_grad():
        ref = model(x)  # original forward (torch.stft/istft, complex)
        got = full(x)  # conv-based, real-arithmetic
    if ref.shape != got.shape:
        print(f"shape mismatch: ref {tuple(ref.shape)} vs got {tuple(got.shape)}")
    snr = snr_db(ref, got)
    print(
        f"gate1 FullSeparator vs model.forward: SNR={snr:.1f} dB  "
        f"max|diff|={float((ref-got).abs().max()):.3e}"
    )
    print("VERDICT:", "PASS" if snr > 60 else "FAIL")  # noqa: PLR2004


def main() -> None:
    global T, TSAMP  # noqa: PLW0603
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--check", action="store_true", help="numerics gate only, no export")
    p.add_argument(
        "--frames",
        type=int,
        default=T,
        help="chunk frame count T (STFT/iSTFT weights are T-independent; only "
        "TSAMP and the static shapes change). The default is the full reference "
        "context the published core uses; a shorter window halves the activation "
        "footprint but loses separation context, measuring ~24 dB at T=501 "
        "against the same track at the default.",
    )
    p.add_argument(
        "--skip-gate",
        action="store_true",
        help="skip the CPU torch-vs-conv numerics gate (T-invariant for the DSP; "
        "validated at T=501). Halves export-time peak RAM at large T.",
    )
    p.add_argument(
        "--fp32",
        action="store_true",
        help="export the full graph in fp32 (skip fp16 conversion) — experiment to "
        "compare quality/perf/VRAM vs the mixed fp16 build.",
    )
    p.add_argument(
        "--fuse-rmsnorm",
        action="store_true",
        help="fuse only the RMSNorm chains to ai.onnx RMSNormalization, leaving "
        "attention alone so --attn-block survives. Unlike the hand-lowered chain, "
        "the WebGPU kernel for this op casts to f32 inside the shader and "
        "accumulates the sum of squares there, so it needs no fp32 island and no "
        "giant cast copies: fp16 in, fp16 out, f32 where it matters.",
    )
    p.add_argument(
        "--all-fp16",
        action="store_true",
        help="convert everything except graph IO to fp16, including the RMSNorm "
        "islands and Softmax that are normally pinned to fp32. Those pins cost "
        "nothing at T=501 but at T=1101 they turn the [T,60,1536] activations "
        "into 387 MiB fp32 tensors plus a cast copy each, which is far past what "
        "an Adreno storage buffer addresses. RMSNormalization then carries "
        f"epsilon={RMSNORM_EPS_FP16:g} instead of {RMSNORM_EPS:g}, which is what "
        "keeps its fp16 reciprocal finite; check parity before trusting the rest, "
        "mean(x^2) can still overflow fp16 in the unfused chain.",
    )
    p.add_argument(
        "--attn-block",
        type=int,
        default=0,
        help="split attention over the query axis into chunks of this many rows "
        "(0 = off). Exact, same FLOPs; it only caps the peak score tensor at "
        "[60,8,block,T] instead of [60,8,T,T], which is what mobile WebGPU "
        "storage buffers cannot bind.",
    )
    p.add_argument(
        "--split-rows",
        type=int,
        default=0,
        help="chunk every feed-forward into this many row groups and unfuse the "
        "qkv projection into three (0 = off). Exact, fp16 included, and the same "
        "FLOPs; it only shortens the two longest matmuls of each layer, which a "
        "mobile GPU otherwise runs as one uninterruptible dispatch each -- the "
        "part of a run that a queue-level flush threshold cannot break up.",
    )
    p.add_argument(
        "--core-only",
        action="store_true",
        help="export the web core (stft_repr -> per-bin masks), NOT the full "
        "STFT/iSTFT graph. Produces the published core that onnxruntime-web runs "
        "with @musetric/fft host-side DSP. The model bakes the mel-band "
        "gather/average tables into the graph.",
    )
    args = p.parse_args()

    T = args.frames
    TSAMP = HOP * (T - 1)
    print(f"frames T={T}  TSAMP={TSAMP}  (~{TSAMP / 44100:.1f}s @44.1k)")

    model = load_model(args.checkpoint, args.config)
    if args.attn_block:
        n = set_attention_block(model, args.attn_block)
        peak = 60 * 8 * args.attn_block * T * 2
        print(
            f"attention blocked at {args.attn_block} query rows over {n} layers; "
            f"peak time-attention score tensor {peak / 1048576:.1f} MiB "
            f"(was {60 * 8 * T * T * 2 / 1048576:.1f} MiB)"
        )
    if args.split_rows:
        feeds, attentions = split_heavy_projections(model, args.split_rows)
        print(
            f"split {feeds} feed-forwards into {args.split_rows} row groups, "
            f"unfused {attentions} qkv projections"
        )
    # warm RotaryEmbedding cache at T (decode path mutates it on first call)
    with torch.no_grad():
        model.net_forward(torch.randn(1, PACKED, T, 2))

    if args.core_only:
        # web-backend core: net_forward only; STFT/iSTFT live host-side in @musetric/fft
        net = Core(model).eval()
        if args.check:
            return
        if args.output is None:
            raise SystemExit("--output required for export")
        export(
            net,
            args.output,
            fp32=args.fp32,
            all_fp16=args.all_fp16,
            fuse_rmsnorm_only=args.fuse_rmsnorm,
            core_only=True,
        )
        return

    full = FullSeparator(model).eval()
    if not args.skip_gate:
        gate1(model, full)

    if args.check:
        return
    if args.output is None:
        raise SystemExit("--output required for export")
    export(
        full,
        args.output,
        fp32=args.fp32,
        all_fp16=args.all_fp16,
        fuse_rmsnorm_only=args.fuse_rmsnorm,
    )


class Core(nn.Module):
    """Core boundary: stft_repr [1,2050,T,2] -> per-bin masks [1,2050,T,2].

    net_forward emits per-band masks [1,1,3958,T,2]; this tail gathers them onto
    the 2050 packed freq bins (fan-in <=2 per bin) and divides by the per-bin band
    count, so the published core needs no host-side band tables. It is a Gather +
    Add + Div formulation (all WebGPU-resident) rather than scatter_add, whose
    ScatterElements has no onnxruntime-web WebGPU kernel and would fall back to CPU.
    """

    def __init__(self, model: MelBandRoformer):
        super().__init__()
        self.model = model
        freq_indices = model.freq_indices.long()
        pair0 = torch.zeros(PACKED, dtype=torch.long)
        pair1 = torch.zeros(PACKED, dtype=torch.long)
        valid0 = torch.zeros(PACKED)
        valid1 = torch.zeros(PACKED)
        seen = torch.zeros(PACKED, dtype=torch.long)
        for band in range(freq_indices.shape[0]):
            b = int(freq_indices[band])
            if seen[b] == 0:
                pair0[b], valid0[b] = band, 1.0
            elif seen[b] == 1:
                pair1[b], valid1[b] = band, 1.0
            else:
                raise RuntimeError(f"mask fan-in > 2 at bin {b}")
            seen[b] += 1
        denom = model.num_bands_per_freq.float().repeat_interleave(2).clamp(min=1e-8)
        self.register_buffer("pair0", pair0)
        self.register_buffer("pair1", pair1)
        self.register_buffer("valid0", valid0.view(1, PACKED, 1, 1))
        self.register_buffer("valid1", valid1.view(1, PACKED, 1, 1))
        self.register_buffer("denom", denom.view(1, PACKED, 1, 1))

    def forward(self, stft_repr: torch.Tensor) -> torch.Tensor:
        masks = self.model.net_forward(stft_repr)
        masks = masks.reshape(1, -1, masks.shape[-2], 2)  # [1,3958,T,2]
        g0 = masks.index_select(1, self.pair0)  # [1,2050,T,2]
        g1 = masks.index_select(1, self.pair1)  # [1,2050,T,2]
        summed = g0 * self.valid0 + g1 * self.valid1
        return (summed / self.denom).float()


def export(  # noqa: C901, PLR0913, PLR0915
    full: nn.Module,
    output: Path,
    fp32: bool = False,
    all_fp16: bool = False,
    fuse_rmsnorm_only: bool = False,
    core_only: bool = False,
) -> None:
    import time  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    # --all-fp16 is the only mode that leaves the normalization in fp16.
    rms_eps = RMSNORM_EPS_FP16 if all_fp16 and not fp32 else RMSNORM_EPS
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(f"{output.stem}_raw.onnx")
    if core_only:
        dummy = torch.randn(1, PACKED, T, 2, dtype=torch.float32) * 0.1
        in_names, out_names = ["stft_repr"], ["masks"]
        print(f"export core graph at stft_repr (1,{PACKED},{T},2) (dynamo, static) ...")
    else:
        dummy = torch.randn(1, 2, TSAMP, dtype=torch.float32) * 0.1
        in_names, out_names = ["raw_audio"], ["vocals"]
        print(f"export full graph at raw_audio (1,2,{TSAMP}) (dynamo, static) ...")

    t0 = time.time()
    torch.onnx.export(
        full,
        (dummy,),
        str(raw),
        input_names=in_names,
        output_names=out_names,
        dynamo=True,
        opset_version=18,
    )
    print(f"  exported in {time.time()-t0:.1f}s")

    import gc  # noqa: PLC0415

    del full
    gc.collect()

    import onnx  # noqa: PLC0415
    from onnxconverter_common import float16  # noqa: PLC0415

    proto = onnx.load(str(raw))
    counts = Counter(n.op_type for n in proto.graph.node)
    audit = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    print("op audit (fp32 export):", audit)
    if "Einsum" in counts:
        raise RuntimeError("Einsum present - matmul attention rewrite did not take")
    for native in ("STFT", "DFT"):
        if native in counts:
            raise RuntimeError(f"{native} op present - DSP did not lower to conv")

    if fuse_rmsnorm_only:
        n = fuse_rmsnorm(proto.graph, rms_eps)
        for op in proto.opset_import:
            if op.domain in ("", "ai.onnx") and op.version < 23:  # noqa: PLR2004
                op.version = 23
        onnx.checker.check_model(proto, full_check=False)
        print(
            f"  fused {n} RMSNorm chains -> RMSNormalization "
            f"(fp16 IO, f32 inside, epsilon={rms_eps:g})"
        )

    if fp32:
        # Full-fp32 variant (experiment): no fp16 conversion, so no fp16<->fp32
        # boundary Casts at all. save_fp16 just writes external data (dtype-
        # agnostic). Doubles activation/weight bytes -> expect slower on the
        # memory-bound majority + higher VRAM (paging risk on small cards).
        print("keeping FP32 (no fp16 conversion) ...")
        save_fp16(proto, output)
    else:
        # Keep the DSP (STFT/iSTFT framing) and the RMSNorm chain (ReduceL2/Clip/
        # Div) + Softmax in fp32; everything else (the NN core MatMul/Gemm) -> fp16.
        # NOTE: a naive fp16 conversion of the RMSNorm chain produces all-NaN
        # output (ReduceL2 = sqrt(sum x^2) overflows fp16 for the large residual
        # stream, and fp16 Clip/Div do not recover it) — measured, parity gate
        # NaN=441000. Those boundary Casts can only be removed by FUSING RMSNorm to
        # SkipSimplifiedLayerNormalization (fp32-internal accumulation), not here.
        # RMSNormalization (the fused replacement) also stays fp32: its eps=1e-12
        # underflows in fp16 (small/zero rows -> 0/0 NaN), and fp32 IO is ~free on
        # this memory-bound graph (per-chunk unchanged). See fuse_rmsnorm's note.
        extra_block = (
            ["Conv", "ConvTranspose", "Pad"]
            if all_fp16
            else [
                "Softmax",
                "ReduceL2",
                "Clip",
                "Div",
                "Conv",
                "ConvTranspose",
                "Pad",
                "RMSNormalization",
            ]
        )
        op_block = list(dict.fromkeys([*float16.DEFAULT_OP_BLOCK_LIST, *extra_block]))
        print(f"converting to FP16 (keep_io_types, block += {extra_block}) ...")
        model16 = float16.convert_float_to_float16(
            proto,
            keep_io_types=True,
            disable_shape_infer=True,
            op_block_list=op_block,
        )
        del proto
        gc.collect()

        if core_only:
            ensure_float32_outputs(model16)
        assert_fp16_epsilon_safe(model16)
        save_fp16(model16, output)
        reloaded = onnx.load(str(output))
        nfix = sanitize_fp16_initializers(reloaded)
        if nfix:
            save_fp16(reloaded, output)
        print(f"  sanitized {nfix} non-finite fp16 weight value(s)")

    raw.unlink(missing_ok=True)
    raw.with_suffix(".onnx.data").unlink(missing_ok=True)
    total = output.stat().st_size + output.with_suffix(".onnx.data").stat().st_size
    print(f"  wrote {output.name} (+data, {total/1e6:.0f} MB)")

    import onnxruntime as ort  # noqa: PLC0415

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        str(output), sess_options=so, providers=["CPUExecutionProvider"]
    )
    print("ORT path-load OK:")
    print("  inputs :", [(i.name, i.shape) for i in sess.get_inputs()])
    print("  outputs:", [(o.name, o.shape) for o in sess.get_outputs()])


def _io_maps(graph):
    producer = {}
    consumers: dict = {}
    for n in graph.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    return producer, consumers


def _sole_consumer(consumers, name):
    cs = consumers.get(name, [])
    return cs[0] if len(cs) == 1 else None


def _rewrite_nodes(graph, remove: set, replace_at: dict) -> None:
    """Drop nodes in `remove`; where a kept node's name is a key of `replace_at`,
    emit the replacement node(s) (a node or a list, in order) at that position
    first (keeps topo order)."""
    nodes = list(graph.node)
    del graph.node[:]
    for n in nodes:
        if n.name in replace_at:
            rep = replace_at[n.name]
            for r in rep if isinstance(rep, list) else [rep]:
                graph.node.append(r)
        if n.name in remove:
            continue
        graph.node.append(n)


def _ensure_ms_opset(model) -> None:
    import onnx  # noqa: PLC0415

    if not any(op.domain == "com.microsoft" for op in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))


def fuse_rmsnorm(graph, epsilon: float) -> int:
    """RMSNorm chain -> standard ONNX RMSNormalization (opset 23, WebGPU kernel,
        fp32-internal accumulation so the fp16 ReduceL2-overflow problem is gone).
        Pattern (fp32): x -> ReduceL2(x) -> Clip(min=eps) -> Expand -> Div(x, .) ->
        Mul(sqrt_dim const) -> Mul(gamma). RMSNormalization(X, scale) = X /
        sqrt(mean(X^2)+eps) * scale, so scale = gamma and the sqrt_dim factor folds
        away (mean vs sum). Caller bumps the onnx opset import to 23.

    epsilon is RMSNORM_EPS (1e-12) while the node stays fp32 and RMSNORM_EPS_FP16
        (1e-8) once it runs in fp16 — see the constants for why each value, and pass
        the one that matches how this graph will be converted. Both are far below the
        1e-6 that corrupts every row with per-element rms <~ 0.03 (~10 dB end-to-end:
        isolated RMSNorm 24 dB vs 139 dB; full graph fp32 48 -> 57 dB).

        The default export keeps the node fp32 (it is in export()'s op_block_list),
        which costs nothing on this memory-bound graph — RMSNormalization accumulates
        the reduction in fp32 internally regardless, and per-chunk time is unchanged
        (682 vs 686 ms). --all-fp16 drops that pin to keep the [T,60,1536] activations
        out of fp32, and then the epsilon is what has to change.

        NOTE: the opset-23 bump pushes ConvTranspose (iSTFT) + Cos/Sin to the CPU on
        the WebGPU EP — accepted on purpose: WebGPU ConvTranspose for the 2048-kernel
        iSTFT measured 200 ms vs CPU 88 ms (2.3x slower on GPU). Fusing instead to
        opset-1 LpNormalization (to keep the iSTFT on WebGPU) was tried and is net
        SLOWER. com.microsoft SkipSimplifiedLayerNormalization needs a full-shape skip;
        SimplifiedLayerNormalization is not a registered op. RMSNormalization stays.
    """
    import onnx  # noqa: PLC0415

    _producer, consumers = _io_maps(graph)
    inits = {t.name for t in graph.initializer}
    remove: set = set()
    replace_at: dict = {}
    count = 0
    for red in [n for n in graph.node if n.op_type == "ReduceL2"]:
        x = red.input[0]
        clip = _sole_consumer(consumers, red.output[0])
        if clip is None or clip.op_type != "Clip":
            continue
        expand = _sole_consumer(consumers, clip.output[0])
        if expand is None or expand.op_type != "Expand":
            continue
        div = _sole_consumer(consumers, expand.output[0])
        if div is None or div.op_type != "Div" or div.input[0] != x:
            continue
        mul1 = _sole_consumer(consumers, div.output[0])
        if mul1 is None or mul1.op_type != "Mul":
            continue
        mul2 = _sole_consumer(consumers, mul1.output[0])
        if mul2 is None or mul2.op_type != "Mul":
            continue
        gamma = next((i for i in mul2.input if i != mul1.output[0]), None)
        if gamma is None or gamma not in inits:
            continue
        rms = onnx.helper.make_node(
            "RMSNormalization",
            [x, gamma],
            [mul2.output[0]],
            name=f"RMSNorm_{count}",
            axis=-1,
            epsilon=epsilon,
        )
        replace_at[red.name] = rms
        remove.update(
            {red.name, clip.name, expand.name, div.name, mul1.name, mul2.name}
        )
        count += 1
    if count:
        _rewrite_nodes(graph, remove, replace_at)
    return count


def save_fp16(model, output: Path) -> None:
    import onnx  # noqa: PLC0415

    output.unlink(missing_ok=True)
    output.with_suffix(".onnx.data").unlink(missing_ok=True)
    onnx.save_model(
        model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output.name + ".data",
        size_threshold=4096,
        convert_attribute=True,
    )


def sanitize_fp16_initializers(model) -> int:
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    fixed = 0
    for t in model.graph.initializer:
        if t.data_type != onnx.TensorProto.FLOAT16:
            continue
        a = numpy_helper.to_array(t).astype(np.float32)
        if not np.isfinite(a).all():
            fixed += int((~np.isfinite(a)).sum())
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
            t.CopyFrom(numpy_helper.from_array(a, t.name))
    return fixed


def assert_fp16_epsilon_safe(model) -> None:
    """Fail the build if a normalization that ended up in fp16 kept an epsilon the
    dtype cannot carry through 1/sqrt(mean(x^2)+eps) — the NaN described at
    RMSNORM_EPS_FP16.
    """
    import onnx  # noqa: PLC0415

    graph = model.graph
    dtypes = {t.name: t.data_type for t in graph.initializer}
    for value in list(graph.value_info) + list(graph.input) + list(graph.output):
        dtypes[value.name] = value.type.tensor_type.elem_type

    bad: list[str] = []
    for node in graph.node:
        fp16 = any(dtypes.get(n) == onnx.TensorProto.FLOAT16 for n in node.output)
        if not fp16:
            continue
        if node.op_type != "RMSNormalization":
            continue
        eps = next((a.f for a in node.attribute if a.name == "epsilon"), None)
        if eps is not None and eps < FP16_RSQRT_FLOOR:
            bad.append(f"{node.name or node.op_type}={eps:g}")
    if bad:
        raise RuntimeError(
            f"{len(bad)} fp16 normalization node(s) carry an epsilon below the fp16 "
            f"reciprocal floor {FP16_RSQRT_FLOOR:g}, which makes every near-zero row "
            f"NaN on a WebGPU kernel: {', '.join(bad[:4])}"
            + (" ..." if len(bad) > 4 else "")  # noqa: PLR2004
        )


def ensure_float32_outputs(model) -> int:
    import onnx  # noqa: PLC0415

    graph = model.graph
    rewritten = 0
    for output in graph.output:
        tensor_type = output.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.FLOAT:
            continue

        original_name = output.name
        cast_input = f"{original_name}_pre_float32_cast"
        producer = next(
            (node for node in graph.node if original_name in node.output),
            None,
        )
        if producer is None:
            raise RuntimeError(f"cannot find producer for graph output {original_name}")
        for i, name in enumerate(producer.output):
            if name == original_name:
                producer.output[i] = cast_input

        graph.node.append(
            onnx.helper.make_node(
                "Cast",
                [cast_input],
                [original_name],
                name=f"{original_name}_to_float32",
                to=onnx.TensorProto.FLOAT,
            )
        )
        tensor_type.elem_type = onnx.TensorProto.FLOAT
        rewritten += 1
    return rewritten


if __name__ == "__main__":
    main()
