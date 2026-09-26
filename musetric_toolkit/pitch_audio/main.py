from pathlib import Path

import numpy as np

from musetric_toolkit.common import envs
from musetric_toolkit.common.logger import send_message
from musetric_toolkit.common.model_files import ensure_model_file
from musetric_toolkit.pitch_audio.tracker import SAMPLE_RATE, load_tracker, track
from musetric_toolkit.pitch_audio.trusted import energy_gate, trusted_mask
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.system_info import ensure_ffmpeg

CONTEXT_SECONDS = 1.0


def _write_csv(
    path: Path,
    time_s: np.ndarray,
    f0_hz: np.ndarray,
    confidence: np.ndarray,
    trusted: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as csv_file:
        csv_file.write("time_s,f0_hz,confidence,trusted\n")
        for index in range(time_s.shape[0]):
            csv_file.write(
                f"{time_s[index]:.4f},{f0_hz[index]:.3f},"
                f"{confidence[index]:.4f},{int(trusted[index])}\n"
            )


def main(args) -> None:
    models_root = Path(args.models_path)
    checkpoint_path = models_root / envs.rmvpe_checkpoint_rel_path
    ensure_model_file(envs.rmvpe_checkpoint_url, checkpoint_path, "RMVPE checkpoint")
    ensure_ffmpeg()
    send_message({"type": "progress", "progress": 0.0})

    hop_samples = max(1, round(SAMPLE_RATE * args.hop_ms / 1000.0))
    audio = read_audio_file(args.audio_path, SAMPLE_RATE, 1)[0]
    total_frames = audio.shape[0] // hop_samples
    first_frame = (
        0
        if args.from_seconds is None
        else int(args.from_seconds * SAMPLE_RATE // hop_samples)
    )
    last_frame = (
        total_frames
        if args.to_seconds is None
        else min(
            total_frames, int(np.ceil(args.to_seconds * SAMPLE_RATE / hop_samples))
        )
    )
    if last_frame <= first_frame:
        raise ValueError("the requested range holds no frames")

    context_frames = int(np.ceil(CONTEXT_SECONDS * SAMPLE_RATE / hop_samples))
    start_frame = max(0, first_frame - context_frames)
    end_frame = min(total_frames, last_frame + context_frames)
    segment = audio[start_frame * hop_samples : end_frame * hop_samples]

    tracker = load_tracker(checkpoint_path, hop_samples)
    send_message({"type": "progress", "progress": 0.2})
    result = track(tracker, segment)
    send_message({"type": "progress", "progress": 0.8})

    frames = end_frame - start_frame
    f0_hz = np.zeros(frames)
    confidence = np.zeros(frames)
    count = min(frames, result.f0_hz.shape[0])
    f0_hz[:count] = result.f0_hz[:count]
    confidence[:count] = result.confidence[:count]
    voiced = energy_gate(segment, SAMPLE_RATE, hop_samples, f0_hz, confidence)
    f0_hz = np.where(voiced, f0_hz, 0.0)
    confidence = np.clip(np.where(voiced, confidence, 0.0), 0.0, 1.0)
    trusted = trusted_mask(f0_hz, confidence)

    keep = slice(first_frame - start_frame, last_frame - start_frame)
    time_s = np.arange(first_frame, last_frame) * (hop_samples / SAMPLE_RATE)
    _write_csv(
        Path(args.result_path), time_s, f0_hz[keep], confidence[keep], trusted[keep]
    )
    send_message({"type": "progress", "progress": 1.0})
