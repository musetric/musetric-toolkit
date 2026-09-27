from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from musetric_toolkit.common import envs
from musetric_toolkit.common.logger import send_message
from musetric_toolkit.common.model_files import ensure_model_files
from musetric_toolkit.parity_audio.case_writer import CaseWriter
from musetric_toolkit.parity_audio.product_stft import analyze, synthesize
from musetric_toolkit.parity_audio.unit_plans import (
    fold,
    normalize_peak,
    unit_window,
    vocals_plan,
)
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.roformer.mel_band_roformer import MelBandRoformer
from musetric_toolkit.separate_audio.roformer_utils import dict_to_namespace

# The vocals half of the separation step: the product feeds the core the STFT of
# an 1100-frame chunk (packages/server/src/analysis/models.rs VOCALS_GEOMETRY),
# multiplies the spectrum by the masks it returns and folds the chunks with the
# host's Hamming weights. The original is the Aname Duality checkpoint in torch.

SAMPLE_RATE = 44100
CHANNELS = 2
N_FFT = 2048
HOP = 441
FRAMES = 1100
CHUNK_SAMPLES = HOP * (FRAMES - 1)


def load_duality(models_path: Path, device: torch.device):
    checkpoint_path = models_path / envs.model_checkpoint_rel_path
    config_path = models_path / envs.model_config_rel_path
    ensure_model_files(checkpoint_path, config_path)
    with config_path.open(encoding="utf-8") as handle:
        raw_config = yaml.load(handle, Loader=yaml.FullLoader)  # noqa: S506
    config = dict_to_namespace(raw_config)
    model = MelBandRoformer(**vars(config.model))
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    return model.to(device).eval(), config


def pack(spectrum: np.ndarray) -> np.ndarray:
    """[channels, frames, bins] -> stft_repr [1, bins * channels, frames, 2]."""
    channels, frames, bins = spectrum.shape
    packed = np.zeros((1, bins * channels, frames, 2))
    for channel in range(channels):
        packed[0, channel::channels, :, 0] = spectrum[channel].T.real
        packed[0, channel::channels, :, 1] = spectrum[channel].T.imag
    return packed


def apply_masks(spectrum: np.ndarray, masks: np.ndarray) -> np.ndarray:
    channels = spectrum.shape[0]
    masked = np.empty_like(spectrum)
    for channel in range(channels):
        mask = (
            masks[0, channel::channels, :, 0] + 1j * masks[0, channel::channels, :, 1]
        )
        masked[channel] = spectrum[channel] * mask.T
    return masked


def torch_masks(model: MelBandRoformer, stft_repr: np.ndarray) -> np.ndarray:
    """The per-bin masks of the original forward: band masks scattered and averaged."""
    device = next(model.parameters()).device
    with torch.inference_mode():
        tensor = torch.from_numpy(stft_repr.astype(np.float32)).to(device)
        masks = model.net_forward(tensor)
        index = model.freq_indices.to(device)
        summed = torch.zeros(
            (1, masks.shape[1], tensor.shape[1], *masks.shape[3:]), device=device
        ).index_add_(2, index, masks)
        denom = model.num_bands_per_freq.to(device).repeat_interleave(CHANNELS)
        averaged = summed / denom.clamp(min=1e-8).view(1, 1, -1, 1, 1)
        return averaged[:, 0].double().cpu().numpy()


def onnx_masks(onnx_path: Path, stft_repr: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )
    (masks,) = session.run(["masks"], {"stft_repr": stft_repr.astype(np.float32)})
    return masks.astype(np.float64)


def author_demix(model: MelBandRoformer, config, mixture: np.ndarray) -> np.ndarray:
    """Vocals by the model's published inference (Music Source Separation Training,
    `demix` in generic mode) with the checkpoint's own chunk, overlap and AMP."""
    device = next(model.parameters()).device
    chunk_size = config.audio.chunk_size
    fade_size = chunk_size // 10
    step = chunk_size // config.inference.num_overlap
    border = chunk_size - step
    batch_size = config.inference.batch_size
    window_base = torch.ones(chunk_size)
    window_base[:fade_size] = torch.linspace(0, 1, fade_size)
    window_base[-fade_size:] = torch.linspace(1, 0, fade_size)

    mix = torch.from_numpy(mixture.astype(np.float32))
    length_init = mix.shape[-1]
    padded = length_init > 2 * border and border > 0
    if padded:
        mix = torch.nn.functional.pad(mix, (border, border), mode="reflect")
    result = torch.zeros(mix.shape)
    counter = torch.zeros(mix.shape)
    batch_data: list[torch.Tensor] = []
    batch_locations: list[tuple[int, int]] = []
    position = 0
    with (
        torch.autocast(device.type, enabled=bool(config.training.use_amp)),
        torch.inference_mode(),
    ):
        while position < mix.shape[1]:
            part = mix[:, position : position + chunk_size]
            chunk_len = part.shape[-1]
            pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
            part = torch.nn.functional.pad(
                part, (0, chunk_size - chunk_len), mode=pad_mode, value=0
            )
            batch_data.append(part)
            batch_locations.append((position, chunk_len))
            position += step
            if len(batch_data) >= batch_size or position >= mix.shape[1]:
                estimated = model(torch.stack(batch_data).to(device)).float().cpu()
                window = window_base.clone()
                if position - step == 0:
                    window[:fade_size] = 1
                elif position >= mix.shape[1]:
                    window[-fade_size:] = 1
                for index, (start, seg_len) in enumerate(batch_locations):
                    result[..., start : start + seg_len] += (
                        estimated[index, ..., :seg_len] * window[..., :seg_len]
                    )
                    counter[..., start : start + seg_len] += window[..., :seg_len]
                batch_data.clear()
                batch_locations.clear()
    sources = torch.nan_to_num(result / counter, nan=0.0)
    if padded:
        sources = sources[..., border:-border]
    return sources.double().numpy()


def run(args, writer: CaseWriter) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, config = load_duality(Path(args.models_path), device)

    source = read_audio_file(args.audio_path, SAMPLE_RATE, CHANNELS).astype(np.float64)
    mixture = normalize_peak(source)
    samples = mixture.shape[1]
    plan = vocals_plan(SAMPLE_RATE, samples, CHUNK_SAMPLES)
    writer.meta.update(
        {
            "sampleRate": SAMPLE_RATE,
            "channels": CHANNELS,
            "samples": samples,
            "units": [[unit.start, unit.length] for unit in plan.units],
        }
    )

    chunks = []
    for index in range(len(plan.units)):
        send_message({"type": "progress", "progress": index / len(plan.units) / 2})
        chunk = unit_window(mixture, plan, index)
        spectrum = analyze(chunk, N_FFT, HOP, FRAMES)
        stft_repr = pack(spectrum)
        masks = torch_masks(model, stft_repr)
        output = synthesize(apply_masks(spectrum, masks), N_FFT, HOP, CHUNK_SAMPLES)
        chunks.append(output)
        if index == 0:
            writer.tensor("unit.input", "reference", chunk)
            writer.tensor("model.input", "reference", stft_repr)
            writer.tensor("model.output", "original", masks)
            writer.tensor(
                "model.output", "onnx-cpu", onnx_masks(Path(args.onnx), stft_repr)
            )
            writer.tensor("unit.output", "reference", output)

    raw_vocals = fold(plan, chunks, samples)
    writer.tensor("step.vocals", "reference", normalize_peak(raw_vocals))
    writer.tensor(
        "step.instrumental", "reference", normalize_peak(mixture - raw_vocals)
    )

    send_message({"type": "progress", "progress": 0.75})
    author = author_demix(model, config, source)
    writer.tensor("step.vocals", "author", normalize_peak(author))
    writer.tensor("step.instrumental", "author", normalize_peak(source - author))
