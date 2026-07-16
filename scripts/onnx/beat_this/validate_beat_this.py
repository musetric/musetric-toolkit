"""Validate the exported Beat This! graph against the torch reference.

For each audio file in --audio it compares beat and downbeat times from:
  * the reference pipeline (`File2Beats`, unpatched torch),
  * the exported graph, run under onnxruntime with the same audio loading,
    log-mel front end, chunking and postprocessing.

Everything except the network is shared, so the reported difference is the
export itself. The WebGPU log-mel that replaces the reference front end in
production is validated on the runtime side, against a spectrogram dumped from
this same torch reference. Run it on the material production feeds the model —
the instrumental stem.

    uv run --group export python scripts/onnx/beat_this/validate_beat_this.py \
      --onnx <export-dir>/beat_this.onnx --audio <audio-dir> \
      --models-path <checkpoint-cache-dir>
"""

# ruff: noqa: T201

import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from musetric_toolkit.rhythm_audio.bpm_estimator import estimate_bpm, estimate_meter

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_RATE = 22050
CHUNK_SIZE = 1500
BORDER_SIZE = 6
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
    return signal


def run_onnx(session: ort.InferenceSession, spect: torch.Tensor):
    """Return postprocessed beat and downbeat times from the exported graph."""
    from beat_this.inference import aggregate_prediction, split_piece  # noqa: PLC0415
    from beat_this.model.postprocessor import Postprocessor  # noqa: PLC0415

    windows, starts = split_piece(spect, CHUNK_SIZE, BORDER_SIZE, True)
    predictions = []
    for window in windows:
        beat, downbeat = session.run(None, {"spect": window.unsqueeze(0).numpy()})
        predictions.append(
            {
                "beat": torch.from_numpy(beat[0]),
                "downbeat": torch.from_numpy(downbeat[0]),
            }
        )
    beat, downbeat = aggregate_prediction(
        predictions,
        starts,
        spect.shape[0],
        CHUNK_SIZE,
        BORDER_SIZE,
        "keep_first",
        "cpu",
    )
    return Postprocessor(type="minimal")(beat, downbeat)


def compare(reference: np.ndarray, exported: np.ndarray) -> str:
    """Describe how two beat-time arrays differ."""
    if len(reference) != len(exported):
        return f"COUNT {len(reference)} vs {len(exported)}"
    if len(reference) == 0:
        return "0 events"
    return f"max dt {np.abs(np.asarray(reference) - np.asarray(exported)).max():.2e}"


def summarize(times: np.ndarray, downbeats: np.ndarray) -> str:
    """Render the CLI-visible rhythm summary for a track."""
    beats = np.asarray(times)
    bpm = estimate_bpm(beats)
    meter = estimate_meter(beats, np.asarray(downbeats))
    return f"bpm {bpm:7.3f} meter {meter}"


def main() -> None:
    """Compare exported graphs and torch reference over a directory of audio."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="path to the exported beat_this.onnx",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="directory of .flac files to validate on",
    )
    parser.add_argument(
        "--models-path",
        required=True,
        help="cache dir for the Beat This! checkpoint download (sets TORCH_HOME)",
    )
    args = parser.parse_args()

    os.environ["TORCH_HOME"] = args.models_path
    from beat_this.inference import File2Beats  # noqa: PLC0415
    from beat_this.preprocessing import LogMelSpect  # noqa: PLC0415

    reference = File2Beats(checkpoint_path="final0", device="cpu")
    log_mel = LogMelSpect(device="cpu")
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    tracks = sorted(args.audio.glob("*.flac"))
    matched = 0
    for track in tracks:
        audio = load_mono(track)
        ref_beats, ref_downbeats = reference(str(track))
        spect = log_mel(torch.tensor(audio, dtype=torch.float32))
        onnx_beats, onnx_downbeats = run_onnx(session, spect)
        beat_diff = compare(ref_beats, onnx_beats)
        downbeat_diff = compare(ref_downbeats, onnx_downbeats)
        exact = "COUNT" not in beat_diff and "COUNT" not in downbeat_diff
        matched += int(exact)
        print(
            f"{track.stem:34s} beats {len(ref_beats):4d} {beat_diff:18s} "
            f"downbeats {len(ref_downbeats):4d} {downbeat_diff:18s} "
            f"{summarize(onnx_beats, onnx_downbeats)}"
        )

    print(f"\nbeat/downbeat count agreement: {matched}/{len(tracks)}")


if __name__ == "__main__":
    main()
