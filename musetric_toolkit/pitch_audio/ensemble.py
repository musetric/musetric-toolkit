from dataclasses import dataclass

import numpy as np
from numba import njit

from musetric_toolkit.pitch_zoo.estimate import FMIN_HZ
from musetric_toolkit.pitch_zoo.pitch_csv import PitchTrack

BIN_CENTS = 10.0
OCTAVE_CENTS = 1200.0
FLOOR = 1e-6
TIE_WEIGHT = 1e-3


@dataclass(frozen=True)
class EnsembleParams:
    voicing_share: float = 0.6
    kernel_cents: float = 30.0
    step_cents: float = 100.0
    step_penalty: float = 0.5
    jump_penalty: float = 8.0
    agree_cents: float = 50.0
    trust_count: int = 3


@dataclass(frozen=True)
class ModelWeights:
    voicing: np.ndarray
    pitch: np.ndarray
    octave_anchors: tuple[int, ...] = ()


REFERENCE_MODELS = ("rmvpe", "crepe", "swiftf0", "fcpe")
REFERENCE_WEIGHTS = ModelWeights(
    voicing=np.array([3.0, 1.0, 1.0, 1.0]),
    pitch=np.array([2.0, 1.0, 1.0, 2.0]),
    octave_anchors=(0, 1),
)


def _cents(f0_hz: np.ndarray) -> np.ndarray:
    return OCTAVE_CENTS * np.log2(np.where(f0_hz > 0.0, f0_hz, FMIN_HZ) / FMIN_HZ)


@njit(cache=True)
def _viterbi(
    emission: np.ndarray, reach: int, penalty: float, jump: float
) -> np.ndarray:
    frames, bins = emission.shape
    score = emission[0].copy()
    back = np.zeros((frames, bins), dtype=np.int32)
    for frame in range(1, frames):
        previous = score.copy()
        top = np.argmax(previous)
        for target in range(bins):
            best_score = -np.inf
            best_source = target
            for source in range(max(0, target - reach), min(bins, target + reach + 1)):
                value = previous[source] - penalty * abs(target - source) / reach
                if value > best_score:
                    best_score = value
                    best_source = source
            if previous[top] - jump > best_score:
                best_score = previous[top] - jump
                best_source = top
            score[target] = best_score + emission[frame, target]
            back[frame, target] = best_source
    path = np.zeros(frames, dtype=np.int64)
    path[frames - 1] = np.argmax(score)
    for frame in range(frames - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


@dataclass(frozen=True)
class RunDecode:
    cents: np.ndarray
    agree_weight: np.ndarray
    agree_count: np.ndarray
    octave_trusted: np.ndarray


def _octave_trusted(
    agree: np.ndarray, below: np.ndarray, voiced: np.ndarray, anchors: tuple
) -> np.ndarray:
    if not anchors:
        return np.zeros(agree.shape[1], dtype=bool)
    rest = np.ones(agree.shape[0], dtype=bool)
    rest[list(anchors)] = False
    return (
        agree[list(anchors)].all(axis=0)
        & below[rest].any(axis=0)
        & (below[rest] | ~voiced[rest]).all(axis=0)
    )


def decode_run(
    cents: np.ndarray, voiced: np.ndarray, model: ModelWeights, params: EnsembleParams
) -> RunDecode:
    weights = model.pitch
    margin = 4.0 * params.kernel_cents
    low = np.floor((cents[voiced].min() - margin) / BIN_CENTS)
    high = np.ceil((cents[voiced].max() + margin) / BIN_CENTS)
    centers = np.arange(low, high + 1) * BIN_CENTS
    distance = centers[None, :, None] - cents[:, None, :]
    bumps = np.exp(-0.5 * (distance / params.kernel_cents) ** 2)
    shaping = np.maximum(weights, TIE_WEIGHT)
    salience = (bumps * (voiced * shaping[:, None])[:, None, :]).sum(axis=0)
    emission = np.log(salience.T / (salience.sum(axis=0)[:, None] + FLOOR) + FLOOR)
    reach = max(1, round(params.step_cents / BIN_CENTS))
    path = centers[_viterbi(emission, reach, params.step_penalty, params.jump_penalty)]
    agree = voiced & (np.abs(cents - path[None, :]) <= params.agree_cents)
    below = voiced & (
        np.abs(cents - path[None, :] + OCTAVE_CENTS) <= params.agree_cents
    )
    agree_weight = (agree * weights[:, None]).sum(axis=0)
    agree_count = agree.sum(axis=0)
    agreeing = np.where(agree, cents, 0.0)
    weighted = (agreeing * weights[:, None]).sum(axis=0) / np.maximum(
        agree_weight, FLOOR
    )
    plain = agreeing.sum(axis=0) / np.maximum(agree_count, 1)
    return RunDecode(
        cents=np.where(
            agree_weight > 0.0, weighted, np.where(agree_count > 0, plain, path)
        ),
        agree_weight=agree_weight,
        agree_count=agree_count,
        octave_trusted=_octave_trusted(agree, below, voiced, model.octave_anchors),
    )


def combine(
    f0_hz: np.ndarray, weights: ModelWeights, params: EnsembleParams
) -> PitchTrack:
    voiced = f0_hz > 0.0
    votes = (voiced * weights.voicing[:, None]).sum(axis=0) / weights.voicing.sum()
    frames = f0_hz.shape[1]
    ensemble_f0 = np.zeros(frames)
    agreement = np.zeros(frames)
    agree_count = np.zeros(frames, dtype=np.int64)
    octave_trusted = np.zeros(frames, dtype=bool)
    cents = _cents(f0_hz)
    edges = np.diff(
        np.concatenate(([0], (votes >= params.voicing_share), [0])).astype(np.int8)
    )
    for start, end in zip(
        np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
    ):
        run = decode_run(cents[:, start:end], voiced[:, start:end], weights, params)
        ensemble_f0[start:end] = FMIN_HZ * np.power(2.0, run.cents / OCTAVE_CENTS)
        agreement[start:end] = run.agree_weight / weights.pitch.sum()
        agree_count[start:end] = run.agree_count
        octave_trusted[start:end] = run.octave_trusted
    return PitchTrack(
        times=np.zeros(frames),
        f0_hz=ensemble_f0,
        confidence=agreement,
        trusted=(ensemble_f0 > 0.0)
        & ((agree_count >= params.trust_count) | octave_trusted),
    )
