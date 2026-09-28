from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musetric_toolkit.common import envs
from musetric_toolkit.common.logger import send_message
from musetric_toolkit.common.model_files import ensure_model_file
from musetric_toolkit.common.paths import list_audio_files
from musetric_toolkit.pitch_audio.tracker import (
    SAMPLE_RATE,
    Tracker,
    load_tracker,
    track,
)
from musetric_toolkit.pitch_audio.trusted import energy_gate, trusted_mask
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.system_info import ensure_ffmpeg

CONTEXT_SECONDS = 1.0
RESULT_NAME = "reference.csv"


@dataclass(frozen=True)
class Reference:
    time_s: np.ndarray
    f0_hz: np.ndarray
    confidence: np.ndarray
    trusted: np.ndarray


def _write_csv(path: Path, reference: Reference) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as csv_file:
        csv_file.write("time_s,f0_hz,confidence,trusted\n")
        for index in range(reference.time_s.shape[0]):
            csv_file.write(
                f"{reference.time_s[index]:.4f},{reference.f0_hz[index]:.3f},"
                f"{reference.confidence[index]:.4f},{int(reference.trusted[index])}\n"
            )


def extract_reference(
    tracker: Tracker, audio: np.ndarray, hop_samples: int, frames: tuple[int, int]
) -> Reference:
    first_frame, last_frame = frames
    total_frames = audio.shape[0] // hop_samples
    context_frames = int(np.ceil(CONTEXT_SECONDS * SAMPLE_RATE / hop_samples))
    start_frame = max(0, first_frame - context_frames)
    end_frame = min(total_frames, last_frame + context_frames)
    segment = audio[start_frame * hop_samples : end_frame * hop_samples]
    result = track(tracker, segment)

    count = end_frame - start_frame
    f0_hz = np.zeros(count)
    confidence = np.zeros(count)
    tracked = min(count, result.f0_hz.shape[0])
    f0_hz[:tracked] = result.f0_hz[:tracked]
    confidence[:tracked] = result.confidence[:tracked]
    voiced = energy_gate(segment, SAMPLE_RATE, hop_samples, f0_hz, confidence)
    f0_hz = np.where(voiced, f0_hz, 0.0)
    confidence = np.clip(np.where(voiced, confidence, 0.0), 0.0, 1.0)
    trusted = trusted_mask(f0_hz, confidence)

    keep = slice(first_frame - start_frame, last_frame - start_frame)
    return Reference(
        time_s=np.arange(first_frame, last_frame) * (hop_samples / SAMPLE_RATE),
        f0_hz=f0_hz[keep],
        confidence=confidence[keep],
        trusted=trusted[keep],
    )


def _frame_range(args, total_frames: int, hop_samples: int) -> tuple[int, int]:
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
    return first_frame, last_frame


def main(args) -> None:
    models_root = Path(args.models_path)
    checkpoint_path = models_root / envs.rmvpe_checkpoint_rel_path
    ensure_model_file(envs.rmvpe_checkpoint_url, checkpoint_path, "RMVPE checkpoint")
    ensure_ffmpeg()
    send_message({"type": "progress", "progress": 0.0})

    hop_samples = max(1, round(SAMPLE_RATE * args.hop_ms / 1000.0))
    tracker = load_tracker(checkpoint_path, hop_samples)
    if args.result_path:
        audio = read_audio_file(args.audio_path, SAMPLE_RATE, 1)[0]
        frames = _frame_range(args, audio.shape[0] // hop_samples, hop_samples)
        reference = extract_reference(tracker, audio, hop_samples, frames)
        _write_csv(Path(args.result_path), reference)
        send_message({"type": "progress", "progress": 1.0})
        return

    audio_paths = list_audio_files(Path(args.audio_path))
    for index, audio_path in enumerate(audio_paths):
        audio = read_audio_file(str(audio_path), SAMPLE_RATE, 1)[0]
        frames = (0, audio.shape[0] // hop_samples)
        reference = extract_reference(tracker, audio, hop_samples, frames)
        _write_csv(Path(args.out_dir) / audio_path.stem / RESULT_NAME, reference)
        send_message({"type": "progress", "progress": (index + 1) / len(audio_paths)})
