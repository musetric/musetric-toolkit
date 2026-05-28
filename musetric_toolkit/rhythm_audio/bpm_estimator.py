import numpy as np

_MIN_BEATS = 2
_MIN_INTERVALS_FOR_IQR = 4


def estimate_bpm(beats: np.ndarray) -> float:
    if len(beats) < _MIN_BEATS:
        return 0.0
    intervals = np.diff(beats)
    if len(intervals) >= _MIN_INTERVALS_FOR_IQR:
        q1, q3 = np.percentile(intervals, [25, 75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        filtered = intervals[(intervals >= lo) & (intervals <= hi)]
        if len(filtered) > 0:
            intervals = filtered
    median_interval = float(np.median(intervals))
    if median_interval <= 0:
        return 0.0
    return 60.0 / median_interval


def estimate_meter(
    beats: np.ndarray,
    downbeats: np.ndarray,
    default: int = 4,
) -> int:
    if len(downbeats) < _MIN_BEATS or len(beats) < _MIN_BEATS:
        return default
    counts: list[int] = []
    epsilon = 1e-3
    for i in range(len(downbeats) - 1):
        start = downbeats[i]
        end = downbeats[i + 1]
        in_bar = int(np.sum((beats >= start - epsilon) & (beats < end - epsilon)))
        if in_bar > 0:
            counts.append(in_bar)
    if not counts:
        return default
    return round(float(np.median(counts)))
