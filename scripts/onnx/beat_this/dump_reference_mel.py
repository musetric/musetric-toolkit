"""Dump reference log-mel spectrograms for the WebGPU front end's parity gate.

The Beat This! export leaves feature extraction out of the ONNX graph: the
runtime computes the log-mel on WebGPU. This writes the torch reference's own
answer so that side can be diffed against it.

For each audio file in --audio it writes, into --out:
  * `<stem>.pcm`   -- the decoded mono waveform, float32 (the shader's input),
  * `<stem>.mel`   -- `LogMelSpect` of that waveform, float32 [frames, mels],
  * `<stem>.json`  -- frame/mel counts.

    uv run python scripts/onnx/beat_this/dump_reference_mel.py \
      --audio <audio-dir> --out <dump-dir> --models-path <checkpoint-cache-dir>
"""

# ruff: noqa: T201

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_RATE = 22050
STEREO_NDIM = 2


def load_mono(path: Path) -> np.ndarray:
    """Load a track the way the reference Audio2Frames does."""
    import soxr  # noqa: PLC0415
    from beat_this.preprocessing import load_audio  # noqa: PLC0415

    signal, sample_rate = load_audio(str(path))
    if signal.ndim == STEREO_NDIM:
        signal = signal.mean(1)
    if sample_rate != SAMPLE_RATE:
        signal = soxr.resample(signal, in_rate=sample_rate, out_rate=SAMPLE_RATE)
    return np.asarray(signal, dtype=np.float32)


def main() -> None:
    """Write reference waveforms and their log-mel spectrograms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--models-path",
        required=True,
        help="cache dir for the Beat This! checkpoint download (sets TORCH_HOME)",
    )
    args = parser.parse_args()

    os.environ["TORCH_HOME"] = args.models_path
    from beat_this.preprocessing import LogMelSpect  # noqa: PLC0415

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_mel = LogMelSpect(device="cpu")

    for track in sorted(args.audio.glob("*.flac")):
        audio = load_mono(track)
        with torch.inference_mode():
            spect = log_mel(torch.from_numpy(audio)).numpy().astype(np.float32)
        (out / f"{track.stem}.pcm").write_bytes(audio.tobytes())
        (out / f"{track.stem}.mel").write_bytes(spect.tobytes())
        (out / f"{track.stem}.json").write_text(
            json.dumps({"frames": int(spect.shape[0]), "mels": int(spect.shape[1])}),
            encoding="utf-8",
        )
        print(f"{track.stem}: samples {len(audio)} spect {spect.shape}")


if __name__ == "__main__":
    main()
