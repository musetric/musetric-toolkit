import math
from dataclasses import dataclass

import numpy as np

from musetric_toolkit.pitch_zoo.pitch_csv import PitchTrack

TOLERANCE_CENTS = 50.0
FLIP_CENTS = 550.0
OCTAVE_CENTS = 1200.0


def cents_between(value: np.ndarray, other: np.ndarray) -> np.ndarray:
    return 1200.0 * np.log2(value / other)


def js_round(value: float) -> int:
    return math.floor(value + 0.5)


def quantile(values: np.ndarray, position: float) -> float:
    if values.shape[0] == 0:
        return 0.0
    ordered = np.sort(values)
    return float(ordered[js_round(position * (ordered.shape[0] - 1))])


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def align_to_reference(reference: PitchTrack, ours: PitchTrack) -> np.ndarray:
    if ours.times.shape[0] == 0:
        return np.zeros(reference.times.shape[0])
    index = np.searchsorted(ours.times, reference.times, side="right") - 1
    return np.where(index >= 0, ours.f0_hz[np.maximum(index, 0)], 0.0)


@dataclass(frozen=True)
class PitchAccuracy:
    frames: int
    scored_frames: int
    rpa: float
    rca: float
    voicing_recall: float
    false_alarm: float
    bias_cents: float
    spread_cents: float
    octave_rate: float
    gap_flip_rate: float
    note_break_rate: float


def _gap_flips(ours: np.ndarray) -> tuple[int, int]:
    voiced = np.flatnonzero(ours > 0.0)
    after_gap = np.flatnonzero(np.diff(voiced) > 1) + 1
    jumps = np.abs(cents_between(ours[voiced[after_gap]], ours[voiced[after_gap - 1]]))
    return after_gap.shape[0], int(np.count_nonzero(jumps > FLIP_CENTS))


def _note_breaks(reference: PitchTrack, ours: np.ndarray) -> tuple[int, int]:
    f0 = reference.f0_hz
    both = reference.trusted[1:] & reference.trusted[:-1] & (f0[1:] > 0) & (f0[:-1] > 0)
    steady = np.zeros(both.shape[0], dtype=bool)
    steady[both] = np.abs(cents_between(f0[1:][both], f0[:-1][both])) <= FLIP_CENTS
    breaks = (ours[1:] > 0.0) != (ours[:-1] > 0.0)
    return int(np.count_nonzero(steady)), int(np.count_nonzero(steady & breaks))


def compare_accuracy(reference: PitchTrack, ours_track: PitchTrack) -> PitchAccuracy:
    ours = align_to_reference(reference, ours_track)
    scored = (reference.f0_hz > 0.0) & reference.trusted
    recalled = scored & (ours > 0.0)
    errors = cents_between(ours[recalled], reference.f0_hz[recalled])
    magnitude = np.abs(errors)
    folded = magnitude % OCTAVE_CENTS
    chroma = np.minimum(folded, OCTAVE_CENTS - folded)
    unvoiced = reference.f0_hz <= 0.0
    gaps, flips = _gap_flips(ours)
    held, breaks = _note_breaks(reference, ours)
    scored_count = int(np.count_nonzero(scored))
    recalled_count = int(np.count_nonzero(recalled))
    return PitchAccuracy(
        frames=reference.times.shape[0],
        scored_frames=scored_count,
        rpa=ratio(np.count_nonzero(magnitude <= TOLERANCE_CENTS), scored_count),
        rca=ratio(np.count_nonzero(chroma <= TOLERANCE_CENTS), scored_count),
        voicing_recall=ratio(recalled_count, scored_count),
        false_alarm=ratio(
            np.count_nonzero(unvoiced & (ours > 0.0)), np.count_nonzero(unvoiced)
        ),
        bias_cents=quantile(errors, 0.5),
        spread_cents=quantile(errors, 0.75) - quantile(errors, 0.25),
        octave_rate=ratio(
            np.count_nonzero(np.abs(magnitude - OCTAVE_CENTS) <= TOLERANCE_CENTS),
            recalled_count,
        ),
        gap_flip_rate=ratio(flips, gaps),
        note_break_rate=ratio(breaks, held),
    )
