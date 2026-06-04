"""Build the FULL separation ONNX graph: raw stereo audio -> vocals, with STFT,
the NN core, mask-apply and iSTFT all inside one graph so onnxruntime-node runs
it end-to-end on the WebGPU EP (1 CPU crossing/chunk, no second device).

STFT/iSTFT are implemented as Conv1d / ConvTranspose1d with precomputed
window x DFT-basis weights (NOT torch.stft/istft, which export to the WebGPU-
unsupported STFT/DFT ops). Mask-apply uses real-arithmetic scatter_add (complex
ops don't export to ONNX). The NN core (`net_forward`) is reused unchanged.

Run from the repo root:
  # numerics gate only (torch vs model.forward, CPU):
  uv run python scripts/onnx/build_full_onnx.py --check \
    --checkpoint tmp/models/MelBandRoformerBigSYHFTV1.ckpt \
    --config tmp/models/config_vocals_mel_band_roformer_big_v1_ft.yaml
  # full export + fp16:
  uv run --group export python scripts/onnx/build_full_onnx.py \
    --checkpoint ... --config ... --output tmp/models/syhft_full_t501.onnx
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
from torch import nn

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

from musetric_toolkit.separate_audio.roformer.attend import Attend
from musetric_toolkit.separate_audio.roformer.mel_band_roformer import MelBandRoformer
from musetric_toolkit.separate_audio.roformer_utils import dict_to_namespace

N_FFT = 2048
HOP = 441
WIN = 2048
# Attention head config (from config_vocals_mel_band_roformer_big_v1_ft.yaml:
# heads=8, dim_head=64 -> dim_inner=512), used by the MHA fusion.
HEADS = 8
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM  # 512
T = 501  # default frame count; override with --frames (mirrors export_chunk.py)
TSAMP = HOP * (T - 1)  # 220500
PAD = N_FFT // 2  # 1024
FREQS = N_FFT // 2 + 1  # 1025
PACKED = FREQS * 2  # 2050


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
        "TSAMP and the static shapes change). Default 501.",
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
        "--fuse",
        action="store_true",
        help="production fusion on the fp32 graph before fp16 conversion: erf-gelu "
        "-> FastGelu, RMSNorm -> RMSNormalization, and the attention core -> "
        "MultiHeadAttention (flash, no T² sim -> removes the VRAM cliff, unblocks "
        "larger T). All WebGPU kernels; cuts dispatch count + boundary Casts.",
    )
    p.add_argument(
        "--core-only",
        action="store_true",
        help="export just net_forward (stft_repr -> masks), NOT the full STFT/iSTFT "
        "graph. Produces the web-backend core (syhft_core_fused_fp16_webgpu.onnx) "
        "that onnxruntime-web runs with @musetric/fft host-side DSP.",
    )
    args = p.parse_args()

    T = args.frames
    TSAMP = HOP * (T - 1)
    print(f"frames T={T}  TSAMP={TSAMP}  (~{TSAMP / 44100:.1f}s @44.1k)")

    model = load_model(args.checkpoint, args.config)
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
        export(net, args.output, fp32=args.fp32, fuse=args.fuse, core_only=True)
        return

    full = FullSeparator(model).eval()
    if not args.skip_gate:
        gate1(model, full)

    if args.check:
        return
    if args.output is None:
        raise SystemExit("--output required for export")
    export(full, args.output, fp32=args.fp32, fuse=args.fuse)


class Core(nn.Module):
    """net_forward boundary: stft_repr [1,2050,T,2] -> masks [1,1,3958,T,2]."""

    def __init__(self, model: MelBandRoformer):
        super().__init__()
        self.model = model

    def forward(self, stft_repr: torch.Tensor) -> torch.Tensor:
        return self.model.net_forward(stft_repr)


def export(  # noqa: C901, PLR0915
    full: nn.Module,
    output: Path,
    fp32: bool = False,
    fuse: bool = False,
    core_only: bool = False,
) -> None:
    import time  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

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

    if fuse:
        ng = fuse_gelu(proto.graph)
        nn_ = fuse_rmsnorm(proto.graph)
        na = fuse_attention(proto.graph)
        _ensure_ms_opset(proto)
        if nn_:
            # RMSNormalization is opset 23; bump the onnx (default) domain import.
            # This bump pushes ConvTranspose (iSTFT) + Cos/Sin to the CPU on the
            # WebGPU EP — which is FINE/FASTER: measured WebGPU ConvTranspose for the
            # 2048-kernel iSTFT shape = 200 ms vs CPU 88 ms (2.3x slower on WebGPU),
            # so the CPU fallback is the better placement. (Tried fusing RMSNorm to
            # opset-1 LpNormalization to keep the iSTFT on WebGPU — net SLOWER, both
            # T=501 and T=1101. Do not re-try; see results-log §iSTFT.)
            for op in proto.opset_import:
                if op.domain in ("", "ai.onnx") and op.version < 23:  # noqa: PLR2004
                    op.version = 23
        onnx.checker.check_model(proto, full_check=False)
        post = Counter(n.op_type for n in proto.graph.node)
        print(
            f"  fused {ng} gelu -> FastGelu, {nn_} RMSNorm -> RMSNormalization, "
            f"{na} attention -> MultiHeadAttention"
        )
        print(
            f"  post-fuse: FastGelu={post.get('FastGelu', 0)} "
            f"RMSNormalization={post.get('RMSNormalization', 0)} "
            f"MultiHeadAttention={post.get('MultiHeadAttention', 0)} "
            f"Softmax={post.get('Softmax', 0)} Erf={post.get('Erf', 0)}"
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
        extra_block = [
            "Softmax",
            "ReduceL2",
            "Clip",
            "Div",
            "Conv",
            "ConvTranspose",
            "Pad",
            "RMSNormalization",
        ]
        op_block = list(dict.fromkeys([*float16.DEFAULT_OP_BLOCK_LIST, *extra_block]))
        print(f"converting to FP16 (keep_io_types, block += {extra_block}) ...")
        model16 = float16.convert_float_to_float16(
            proto, keep_io_types=True, disable_shape_infer=True, op_block_list=op_block
        )
        del proto
        gc.collect()

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


def fuse_gelu(graph) -> int:
    """erf-gelu chain -> com.microsoft FastGelu. Pattern (fp32, no casts):
    z -> Div(z, sqrt2) -> Erf -> Add(., 1) -> Mul(., 0.5) -> Mul(., z).

    NOTE: FastGelu is the tanh approximation, not the exact erf GELU the model
    uses. Measured: swapping it for the exact ONNX Gelu(approximate="none") (also
    a WebGPU kernel) leaves parity vs torch UNCHANGED (45.64 -> 45.67 dB). The
    ~45 dB floor is fp16 quantization (~-45 dB) + MHA-flash vs torch attention
    (~-47 dB), not the gelu approximation — so FastGelu stays (no reason to churn).
    """
    import onnx  # noqa: PLC0415

    producer, consumers = _io_maps(graph)
    remove: set = set()
    replace_at: dict = {}
    count = 0
    for erf in [n for n in graph.node if n.op_type == "Erf"]:
        div = producer.get(erf.input[0])
        if div is None or div.op_type != "Div":
            continue
        z = div.input[0]
        add1 = _sole_consumer(consumers, erf.output[0])
        if add1 is None or add1.op_type != "Add":
            continue
        mul_half = _sole_consumer(consumers, add1.output[0])
        if mul_half is None or mul_half.op_type != "Mul":
            continue
        mul_z = _sole_consumer(consumers, mul_half.output[0])
        if mul_z is None or mul_z.op_type != "Mul":
            continue
        if z not in mul_z.input:
            continue
        fg = onnx.helper.make_node(
            "FastGelu",
            [z],
            [mul_z.output[0]],
            domain="com.microsoft",
            name=f"FastGelu_{count}",
        )
        replace_at[div.name] = fg
        remove.update({div.name, erf.name, add1.name, mul_half.name, mul_z.name})
        count += 1
    if count:
        _rewrite_nodes(graph, remove, replace_at)
    return count


def fuse_rmsnorm(graph) -> int:
    """RMSNorm chain -> standard ONNX RMSNormalization (opset 23, WebGPU kernel,
    fp32-internal accumulation so the fp16 ReduceL2-overflow problem is gone).
    Pattern (fp32): x -> ReduceL2(x) -> Clip(min=eps) -> Expand -> Div(x, .) ->
    Mul(sqrt_dim const) -> Mul(gamma). RMSNormalization(X, scale) = X /
    sqrt(mean(X^2)+eps) * scale, so scale = gamma and the sqrt_dim factor folds
    away (mean vs sum). Caller bumps the onnx opset import to 23.

    epsilon=1e-12 (NOT 1e-6): the model's RMSNorm is F.normalize(x, dim=-1) =
    x/max(||x||_2, 1e-12), i.e. a tiny floor on the L2 norm. RMSNormalization adds
    epsilon to mean(x^2) instead, so for a row with per-element rms r the denom is
    sqrt(r^2 + eps). With eps=1e-6 any row with r <~ 0.03 is corrupted (denom off
    by up to sqrt(2)); such small-magnitude rows DO occur (BandSplit / early
    features) and cost ~10 dB end-to-end (measured: isolated RMSNorm 24 dB vs 139
    dB; full graph fp32 48 -> 57 dB). 1e-12 matches F.normalize. NOTE: 1e-12
    underflows in fp16 (-> 0/0 NaN), so this node must stay fp32 in the fp16
    conversion (it is in export()'s op_block_list); RMSNormalization accumulates
    the reduction in fp32 internally regardless, so keeping fp32 IO is ~free
    (memory-bound graph; measured per-chunk unchanged, 682 vs 686 ms).

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
            epsilon=1e-12,
        )
        replace_at[red.name] = rms
        remove.update(
            {red.name, clip.name, expand.name, div.name, mul1.name, mul2.name}
        )
        count += 1
    if count:
        _rewrite_nodes(graph, remove, replace_at)
    return count


def fuse_attention(graph) -> int:
    """Attention core MatMul(q,kᵀ) -> Mul(scale) -> Softmax -> MatMul(.,v) ->
    com.microsoft MultiHeadAttention (flash-style WebGPU kernel: no T² sim
    materialization -> removes the VRAM paging cliff, unblocking larger T).

    The model runs attention in [b,h,n,d] (BNSH) with a per-head sigmoid gate
    after it; MHA is BSD [b,n,hidden]. Convert q/k/v BNSH->BSD on the way in, run
    MHA, convert its output BSD->BNSH (output name kept = old matmul output), and
    leave the existing gating + to_out untouched. Rotary stays applied to q/k.
    """
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    producer, consumers = _io_maps(graph)
    inits = {t.name: t for t in graph.initializer}
    bsd_shape, bnhd_shape, have_shapes = "mha_bsd_shape", "mha_bnhd_shape", False
    remove: set = set()
    replace_at: dict = {}
    count = 0
    for sm in [n for n in graph.node if n.op_type == "Softmax"]:
        mul_scale = producer.get(sm.input[0])
        if mul_scale is None or mul_scale.op_type != "Mul":
            continue
        sim = producer.get(mul_scale.input[0])
        if sim is None or sim.op_type != "MatMul":
            continue
        q = sim.input[0]
        k_t = producer.get(sim.input[1])
        if k_t is None or k_t.op_type != "Transpose":
            continue
        k = k_t.input[0]
        matmul2 = _sole_consumer(consumers, sm.output[0])
        if matmul2 is None or matmul2.op_type != "MatMul":
            continue
        v = matmul2.input[1]
        attn_out = matmul2.output[0]
        scale_name = next((i for i in mul_scale.input if i in inits), None)
        if scale_name is None:
            continue
        scale_val = float(numpy_helper.to_array(inits[scale_name]).reshape(-1)[0])

        if not have_shapes:
            # batch is packed (b*freq or b*time, not 1) -> leading 0 = "keep the
            # input's batch dim" (Reshape allowzero=0 default); -1 infers seq.
            graph.initializer.append(
                numpy_helper.from_array(
                    np.array([0, -1, HIDDEN], dtype=np.int64), bsd_shape
                )
            )
            graph.initializer.append(
                numpy_helper.from_array(
                    np.array([0, -1, HEADS, HEAD_DIM], dtype=np.int64), bnhd_shape
                )
            )
            have_shapes = True

        p = f"mha{count}"
        nodes = []
        bsd = {}
        for tag, t in (("q", q), ("k", k), ("v", v)):
            nodes.append(
                onnx.helper.make_node(
                    "Transpose",
                    [t],
                    [f"{p}_{tag}_bnhd"],
                    name=f"{p}_{tag}_tr",
                    perm=[0, 2, 1, 3],
                )
            )
            nodes.append(
                onnx.helper.make_node(
                    "Reshape",
                    [f"{p}_{tag}_bnhd", bsd_shape],
                    [f"{p}_{tag}_bsd"],
                    name=f"{p}_{tag}_rs",
                )
            )
            bsd[tag] = f"{p}_{tag}_bsd"
        nodes.append(
            onnx.helper.make_node(
                "MultiHeadAttention",
                [bsd["q"], bsd["k"], bsd["v"]],
                [f"{p}_out_bsd"],
                name=f"{p}_mha",
                domain="com.microsoft",
                num_heads=HEADS,
                scale=scale_val,
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Reshape",
                [f"{p}_out_bsd", bnhd_shape],
                [f"{p}_out_bnhd"],
                name=f"{p}_out_rs",
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Transpose",
                [f"{p}_out_bnhd"],
                [attn_out],
                name=f"{p}_out_tr",
                perm=[0, 2, 1, 3],
            )
        )

        replace_at[sim.name] = nodes
        remove.update({sim.name, mul_scale.name, sm.name, matmul2.name, k_t.name})
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


if __name__ == "__main__":
    main()
