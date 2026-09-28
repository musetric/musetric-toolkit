import sys
from pathlib import Path

import numpy as np

from musetric_toolkit.pitch_zoo.accuracy import cents_between
from musetric_toolkit.pitch_zoo.pitch_csv import read_pitch_csv
from musetric_toolkit.pitch_zoo.score import list_tracks

GROSS_CENTS = 100.0
LAG_STEP_MS = 0.5


def lag_errors(
    truth_times: np.ndarray,
    truth_f0: np.ndarray,
    times: np.ndarray,
    f0_hz: np.ndarray,
    lags: np.ndarray,
) -> list[np.ndarray]:
    log_truth = np.log2(np.where(truth_f0 > 0.0, truth_f0, 1.0))
    voiced_truth = (truth_f0 > 0.0).astype(np.float64)
    errors = []
    for lag in lags:
        at = times + lag
        coverage = np.interp(at, truth_times, voiced_truth)
        shifted = np.power(2.0, np.interp(at, truth_times, log_truth))
        usable = (coverage >= 1.0) & (f0_hz > 0.0)
        cents = np.abs(cents_between(f0_hz[usable], shifted[usable]))
        errors.append(cents[cents < GROSS_CENTS])
    return errors


def main(args) -> None:
    lags = np.arange(-args.range_ms, args.range_ms + LAG_STEP_MS / 2, LAG_STEP_MS)
    tracks = list_tracks(Path(args.tracks_dir), args.truth)
    lines = [f"{len(tracks)} tracks against `{args.truth}`, lag in ms of the truth\n"]
    for model in args.models:
        pooled: list[list[np.ndarray]] = [[] for _ in lags]
        for track in tracks:
            truth = read_pitch_csv(track / f"{args.truth}.csv")
            candidate = read_pitch_csv(track / f"{model}.csv")
            errors = lag_errors(
                truth.times, truth.f0_hz, candidate.times, candidate.f0_hz, lags / 1000
            )
            for index, error in enumerate(errors):
                pooled[index].append(error)
        medians = np.array([float(np.median(np.concatenate(part))) for part in pooled])
        best = int(np.argmin(medians))
        zero = int(np.argmin(np.abs(lags)))
        lines.append(
            f"{model}: best {lags[best]:+.1f} ms ({medians[best]:.2f} c), "
            f"at 0 ms {medians[zero]:.2f} c\n"
        )
    sys.__stdout__.buffer.write("".join(lines).encode("utf-8"))
    sys.__stdout__.flush()
