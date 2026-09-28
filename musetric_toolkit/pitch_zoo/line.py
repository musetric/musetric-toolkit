from dataclasses import dataclass

import numpy as np

from musetric_toolkit.pitch_zoo.accuracy import (
    align_to_reference,
    cents_between,
    js_round,
    quantile,
)
from musetric_toolkit.pitch_zoo.pitch_csv import PitchTrack

WRONG_CENTS = 50.0
STEP_CENTS = 50.0
STILL_CENTS = 25.0
STEP_LAG_MS = 10.0
LEAP_CENTS = 300.0
JERKY_HALO_MS = 25.0
RATTLE_CENTS = 10.0
RATTLE_SMOOTH_MS = 50.0
RATTLE_WINDOW_MS = 100.0
RATTLE_OUTLIER_CENTS = 100.0
RATTLE_MIN_FRAMES = 5
HOLE_MS = 30.0
LONG_HOLE_MS = 150.0
SPECK_MS = 50.0
PHANTOM_MS = 30.0
ONSET_GAP_MS = 100.0
ONSET_SEARCH_MS = 300.0
WINDOW_SECONDS = 5.0
WINDOW_STEP_SECONDS = 1.0
WINDOW_MIN_SINGING_SECONDS = 1.5

NONE, CLEAN, MISSING, WRONG, JERKY, RATTLING = range(6)


@dataclass(frozen=True)
class PitchLine:
    times: np.ndarray
    reference: np.ndarray
    ours: np.ndarray
    hop_seconds: float
    singing: np.ndarray
    voiced: np.ndarray
    error: np.ndarray


def create_line(reference: PitchTrack, ours_track: PitchTrack) -> PitchLine:
    ours = align_to_reference(reference, ours_track)
    times = reference.times
    both = (reference.f0_hz > 0.0) & (ours > 0.0)
    error = np.full(times.shape[0], np.nan)
    error[both] = cents_between(ours[both], reference.f0_hz[both])
    return PitchLine(
        times=times,
        reference=reference.f0_hz,
        ours=ours,
        hop_seconds=float(times[1] - times[0]) if times.shape[0] > 1 else 0.0,
        singing=reference.trusted & (reference.f0_hz > 0.0),
        voiced=ours > 0.0,
        error=error,
    )


def frames_of(line: PitchLine, ms: float) -> int:
    return max(1, js_round(ms / 1000.0 / line.hop_seconds))


def runs_of(mask: np.ndarray) -> np.ndarray:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.stack((np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)), 1)


def milliseconds_of(line: PitchLine, runs: np.ndarray) -> np.ndarray:
    return (runs[:, 1] - runs[:, 0]) * line.hop_seconds * 1000.0


def find_jerks(line: PitchLine) -> tuple[np.ndarray, np.ndarray]:
    lag = frames_of(line, STEP_LAG_MS)
    steps = np.zeros(line.times.shape[0])
    now = np.arange(lag, steps.shape[0])
    before = now - lag
    usable = (
        line.voiced[now]
        & line.voiced[before]
        & (line.reference[now] > 0.0)
        & (line.reference[before] > 0.0)
    )
    now, before = now[usable], before[usable]
    moved = np.abs(cents_between(line.ours[now], line.ours[before]))
    voice_moved = np.abs(cents_between(line.reference[now], line.reference[before]))
    steps[now] = np.where(
        (moved > STEP_CENTS) & (voice_moved < STILL_CENTS), moved, 0.0
    )
    runs = runs_of(steps > 0.0)
    cents = np.array([steps[start:end].max() for start, end in runs])
    return np.stack((runs[:, 0] - lag, runs[:, 1]), 1), cents


def mark_jerky(line: PitchLine, jerks: np.ndarray) -> np.ndarray:
    halo = frames_of(line, JERKY_HALO_MS)
    jerky = np.zeros(line.times.shape[0], dtype=bool)
    for start, end in jerks:
        jerky[max(0, start - halo) : min(jerky.shape[0], end + halo)] = True
    return jerky


def _window_mean(
    values: np.ndarray, usable: np.ndarray, half: int
) -> tuple[np.ndarray, np.ndarray]:
    length = values.shape[0]
    total = np.concatenate(([0.0], np.cumsum(np.where(usable, values, 0.0))))
    counts = np.concatenate(([0.0], np.cumsum(usable.astype(np.float64))))
    index = np.arange(length)
    start = np.maximum(0, index - half)
    end = np.minimum(length, index + half + 1)
    count = counts[end] - counts[start]
    mean = np.full(length, np.nan)
    some = count > 0
    mean[some] = (total[end] - total[start])[some] / count[some]
    return mean, count


@dataclass(frozen=True)
class PitchRattle:
    fast_squared: np.ndarray
    usable: np.ndarray
    local: np.ndarray


def measure_rattle(line: PitchLine) -> PitchRattle:
    usable = np.isfinite(line.error) & (np.abs(line.error) <= RATTLE_OUTLIER_CENTS)
    smooth, _ = _window_mean(line.error, usable, frames_of(line, RATTLE_SMOOTH_MS / 2))
    fast_squared = np.where(usable, (line.error - smooth) ** 2, 0.0)
    mean, count = _window_mean(
        fast_squared, usable, frames_of(line, RATTLE_WINDOW_MS / 2)
    )
    local = np.where(usable & (count >= RATTLE_MIN_FRAMES), np.sqrt(mean), np.nan)
    return PitchRattle(fast_squared=fast_squared, usable=usable, local=local)


def count_phantoms(line: PitchLine) -> tuple[int, int]:
    voice = line.reference > 0.0
    runs = runs_of(line.voiced & ~voice)
    runs = runs[milliseconds_of(line, runs) >= PHANTOM_MS]
    padded = np.concatenate(([False], voice, [False]))
    tails = int(np.count_nonzero(padded[runs[:, 0]] | padded[runs[:, 1] + 1]))
    return runs.shape[0] - tails, tails


def measure_onsets(line: PitchLine) -> tuple[np.ndarray, int]:
    gap = frames_of(line, ONSET_GAP_MS)
    search = frames_of(line, ONSET_SEARCH_MS)
    right = np.abs(np.nan_to_num(line.error, nan=np.inf)) <= WRONG_CENTS
    delays = []
    missed = 0
    previous_end = -gap
    for start, end in runs_of(line.singing):
        if start - previous_end >= gap:
            found = np.flatnonzero(right[start : min(right.shape[0], start + search)])
            if found.shape[0] == 0:
                missed += 1
            else:
                delays.append(found[0] * line.hop_seconds * 1000.0)
        previous_end = end
    return np.array(delays), missed


def classify(line: PitchLine, jerky: np.ndarray, rattle: PitchRattle) -> np.ndarray:
    classes = np.full(line.times.shape[0], CLEAN, dtype=np.int8)
    classes[np.nan_to_num(rattle.local) > RATTLE_CENTS] = RATTLING
    classes[jerky] = JERKY
    classes[np.abs(np.nan_to_num(line.error)) > WRONG_CENTS] = WRONG
    classes[~line.voiced] = MISSING
    classes[~line.singing] = NONE
    return classes


@dataclass(frozen=True)
class LineOverview:
    singing_minutes: float
    clean_share: float
    missing_share: float
    wrong_share: float
    jerky_share: float
    rattling_share: float
    rattle_cents: float
    rattle_p90_cents: float
    steps_per_minute: float
    leaps_per_minute: float
    holes_per_minute: float
    long_holes_per_minute: float
    specks_per_minute: float
    islands_per_minute: float
    tails_per_minute: float
    onset_delay_ms: float
    onset_delay_p90_ms: float
    missed_onset_share: float


def summarize(
    line: PitchLine, classes: np.ndarray, jerk_cents: np.ndarray, rattle: PitchRattle
) -> LineOverview:
    singing_frames = int(np.count_nonzero(line.singing))
    minutes = singing_frames * line.hop_seconds / 60.0

    def per_minute(count: int) -> float:
        return count / minutes if minutes > 0 else 0.0

    def share(value: int) -> float:
        return (
            np.count_nonzero(classes == value) / singing_frames
            if singing_frames
            else 0.0
        )

    holes = milliseconds_of(line, runs_of(line.singing & ~line.voiced))
    voiced_runs = milliseconds_of(line, runs_of(line.voiced))
    islands, tails = count_phantoms(line)
    delays, missed = measure_onsets(line)
    fast = line.singing & rattle.usable
    local = rattle.local[line.singing & np.isfinite(rattle.local)]
    return LineOverview(
        singing_minutes=minutes,
        clean_share=share(CLEAN),
        missing_share=share(MISSING),
        wrong_share=share(WRONG),
        jerky_share=share(JERKY),
        rattling_share=share(RATTLING),
        rattle_cents=(
            float(np.sqrt(rattle.fast_squared[fast].mean())) if fast.any() else 0.0
        ),
        rattle_p90_cents=quantile(local, 0.9),
        steps_per_minute=per_minute(int(np.count_nonzero(jerk_cents <= LEAP_CENTS))),
        leaps_per_minute=per_minute(int(np.count_nonzero(jerk_cents > LEAP_CENTS))),
        holes_per_minute=per_minute(int(np.count_nonzero(holes >= HOLE_MS))),
        long_holes_per_minute=per_minute(int(np.count_nonzero(holes >= LONG_HOLE_MS))),
        specks_per_minute=per_minute(int(np.count_nonzero(voiced_runs < SPECK_MS))),
        islands_per_minute=per_minute(islands),
        tails_per_minute=per_minute(tails),
        onset_delay_ms=quantile(delays, 0.5),
        onset_delay_p90_ms=quantile(delays, 0.9),
        missed_onset_share=missed / max(1, missed + delays.shape[0]),
    )


@dataclass(frozen=True)
class LineWindow:
    from_seconds: float
    to_seconds: float
    singing_seconds: float
    bad_share: float
    missing_share: float
    wrong_share: float
    jerky_share: float
    rattling_share: float


def find_worst_windows(
    line: PitchLine, classes: np.ndarray, count: int
) -> list[LineWindow]:
    size = js_round(WINDOW_SECONDS / line.hop_seconds)
    step = js_round(WINDOW_STEP_SECONDS / line.hop_seconds)
    windows = []
    for start in range(0, classes.shape[0] - size + 1, step):
        counts = np.bincount(classes[start : start + size], minlength=6)
        singing = int(counts[1:].sum())
        if singing * line.hop_seconds < WINDOW_MIN_SINGING_SECONDS:
            continue
        from_seconds = float(line.times[start])
        windows.append(
            LineWindow(
                from_seconds=from_seconds,
                to_seconds=from_seconds + WINDOW_SECONDS,
                singing_seconds=singing * line.hop_seconds,
                bad_share=1.0 - counts[CLEAN] / singing,
                missing_share=counts[MISSING] / singing,
                wrong_share=counts[WRONG] / singing,
                jerky_share=counts[JERKY] / singing,
                rattling_share=counts[RATTLING] / singing,
            )
        )
    picked: list[LineWindow] = []
    for window in sorted(windows, key=lambda entry: -entry.bad_share):
        apart = all(
            window.to_seconds <= other.from_seconds
            or window.from_seconds >= other.to_seconds
            for other in picked
        )
        if apart and len(picked) < count:
            picked.append(window)
    return picked


@dataclass(frozen=True)
class LineResult:
    overview: LineOverview
    worst: list[LineWindow]


def compare_line(
    reference: PitchTrack, ours: PitchTrack, worst_count: int
) -> LineResult:
    line = create_line(reference, ours)
    jerks, jerk_cents = find_jerks(line)
    rattle = measure_rattle(line)
    classes = classify(line, mark_jerky(line, jerks), rattle)
    return LineResult(
        overview=summarize(line, classes, jerk_cents, rattle),
        worst=find_worst_windows(line, classes, worst_count),
    )
