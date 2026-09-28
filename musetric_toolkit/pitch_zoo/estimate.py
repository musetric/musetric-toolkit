from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

SAMPLE_RATE = 16000
FMIN_HZ = 50.0
FMAX_HZ = 1100.0
BRIDGE_CENTS = 100.0


@dataclass(frozen=True)
class ModelContext:
    models_path: Path
    device: torch.device


@dataclass(frozen=True)
class PitchEstimate:
    times: np.ndarray
    f0_hz: np.ndarray
    confidence: np.ndarray


class PitchModel(Protocol):
    def estimate(self, audio: np.ndarray) -> PitchEstimate: ...


def _log2(values: np.ndarray) -> np.ndarray:
    return np.log2(np.where(values > 0.0, values, 1.0))


def to_grid(estimate: PitchEstimate, times: np.ndarray) -> PitchEstimate:
    source = estimate.times
    right = np.clip(np.searchsorted(source, times), 1, source.shape[0] - 1)
    left = right - 1
    span = source[right] - source[left]
    weight = np.clip((times - source[left]) / np.where(span > 0.0, span, 1.0), 0.0, 1.0)
    f0_left = estimate.f0_hz[left]
    f0_right = estimate.f0_hz[right]
    nearest = np.where(times - source[left] < source[right] - times, left, right)
    f0_hz = estimate.f0_hz[nearest].astype(np.float64)
    both = (f0_left > 0.0) & (f0_right > 0.0)
    step = np.abs(1200.0 * (_log2(f0_right) - _log2(f0_left)))
    bridge = both & (step <= BRIDGE_CENTS)
    blended = np.power(2.0, _log2(f0_left) * (1.0 - weight) + _log2(f0_right) * weight)
    f0_hz[bridge] = blended[bridge]
    confidence = (
        estimate.confidence[left] * (1.0 - weight) + estimate.confidence[right] * weight
    )
    step_seconds = float(np.median(np.diff(source))) if source.shape[0] > 1 else 0.0
    outside = (times < source[0] - step_seconds) | (times > source[-1] + step_seconds)
    f0_hz[outside] = 0.0
    confidence = np.where(outside, 0.0, confidence)
    return PitchEstimate(times=times, f0_hz=f0_hz, confidence=confidence)
