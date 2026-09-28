from pathlib import Path

import numpy as np
import pyworld
import soundfile

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.common.paths import list_audio_files
from musetric_toolkit.pitch_zoo.pitch_csv import (
    PitchTrack,
    read_pitch_csv,
    write_pitch_csv,
)
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.system_info import ensure_ffmpeg

SAMPLE_RATE = 48000
PEAK = 0.99


def resynthesize(audio: np.ndarray, curve: PitchTrack, hop_ms: float) -> np.ndarray:
    signal = audio.astype(np.float64)
    frames = int(signal.shape[0] / SAMPLE_RATE * 1000.0 / hop_ms) + 1
    times = np.arange(frames) * (hop_ms / 1000.0)
    f0_hz = np.zeros(frames)
    count = min(frames, curve.f0_hz.shape[0])
    f0_hz[:count] = curve.f0_hz[:count]
    envelope = pyworld.cheaptrick(signal, f0_hz, times, SAMPLE_RATE)
    aperiodicity = pyworld.d4c(signal, f0_hz, times, SAMPLE_RATE)
    voice = pyworld.synthesize(f0_hz, envelope, aperiodicity, SAMPLE_RATE, hop_ms)
    voice = voice[: signal.shape[0]]
    peak = float(np.max(np.abs(voice))) if voice.shape[0] else 0.0
    return voice * (PEAK / peak) if peak > PEAK else voice


def main(args) -> None:
    ensure_ffmpeg()
    set_dir = Path(args.data_dir) / args.name
    audio_paths = list_audio_files(Path(args.audio_path))
    for index, audio_path in enumerate(audio_paths):
        curve = read_pitch_csv(
            Path(args.f0_dir) / audio_path.stem / f"{args.f0_name}.csv"
        )
        audio = read_audio_file(str(audio_path), SAMPLE_RATE, 1)[0]
        voice = resynthesize(audio, curve, args.hop_ms)
        out_path = set_dir / "audio" / f"{audio_path.stem}.flac"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(out_path, voice.astype(np.float32), SAMPLE_RATE, "PCM_24")
        truth = PitchTrack(
            times=curve.times,
            f0_hz=curve.f0_hz,
            confidence=(curve.f0_hz > 0.0).astype(np.float64),
            trusted=np.ones(curve.times.shape[0], dtype=bool),
        )
        write_pitch_csv(
            set_dir / "tracks" / audio_path.stem / "truth.csv",
            truth,
            with_trusted=True,
        )
        send_message({"type": "progress", "progress": (index + 1) / len(audio_paths)})
