import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.pitch_zoo.estimate import SAMPLE_RATE, ModelContext, to_grid
from musetric_toolkit.pitch_zoo.lag import lag_errors
from musetric_toolkit.pitch_zoo.registry import ZOO_MODELS

DURATION_SECONDS = 4.0
ONSET_SECONDS = 0.5
OFFSET_SECONDS = 3.5
SCORED_FROM_SECONDS = 0.7
SCORED_TO_SECONDS = 3.3
HARMONICS = 9
ROLLOFF = 0.7
PEAK = 0.3
NOISE = 0.002
VIBRATO_HZ = 6.0
VIBRATO_CENTS = 80.0
GLIDE_LEG_SECONDS = 0.75
GLIDE_CENTS = 1200.0
HOP_SECONDS = 0.005
TRUTH_STEP_SECONDS = 0.001
LAG_RANGE_MS = 40.0
LAG_STEP_MS = 0.5
ALIGNED_MS = 2.5


@dataclass(frozen=True)
class Signal:
    name: str
    base_hz: float
    vibrato: bool


SIGNALS = (
    Signal("vibrato 110 Hz", 110.0, vibrato=True),
    Signal("vibrato 250 Hz", 250.0, vibrato=True),
    Signal("vibrato 600 Hz", 600.0, vibrato=True),
    Signal("glide 180-360 Hz", 180.0, vibrato=False),
    Signal("glide 400-800 Hz", 400.0, vibrato=False),
)


def f0_curve(signal: Signal, times: np.ndarray) -> np.ndarray:
    local = times - ONSET_SECONDS
    if signal.vibrato:
        cents = VIBRATO_CENTS * np.sin(2.0 * np.pi * VIBRATO_HZ * local)
    else:
        cents = GLIDE_CENTS * np.abs((local / GLIDE_LEG_SECONDS) % 2.0 - 1.0)
    active = (times >= ONSET_SECONDS) & (times < OFFSET_SECONDS)
    return np.where(active, signal.base_hz * np.power(2.0, cents / 1200.0), 0.0)


def synthesize(signal: Signal) -> np.ndarray:
    times = np.arange(int(DURATION_SECONDS * SAMPLE_RATE)) / SAMPLE_RATE
    f0_hz = f0_curve(signal, times)
    phase = 2.0 * np.pi * np.cumsum(f0_hz) / SAMPLE_RATE
    voice = sum(
        np.sin(harmonic * phase) * ROLLOFF**harmonic
        for harmonic in range(1, HARMONICS + 1)
    ) * (f0_hz > 0.0)
    noise = np.random.default_rng(0).standard_normal(times.shape[0])
    return (PEAK * voice / np.max(np.abs(voice)) + NOISE * noise).astype(np.float32)


@dataclass(frozen=True)
class Alignment:
    lag_ms: float
    cents: float
    cents_at_zero: float
    voiced_share: float


def align_model(estimate, signal: Signal, audio: np.ndarray) -> Alignment:
    grid = np.arange(int(DURATION_SECONDS / HOP_SECONDS)) * HOP_SECONDS
    track = to_grid(estimate(audio), grid)
    scored = (grid >= SCORED_FROM_SECONDS) & (grid <= SCORED_TO_SECONDS)
    f0_hz = np.where(scored, track.f0_hz, 0.0)
    truth_times = np.arange(0.0, DURATION_SECONDS, TRUTH_STEP_SECONDS)
    lags_ms = np.arange(-LAG_RANGE_MS, LAG_RANGE_MS + LAG_STEP_MS / 2, LAG_STEP_MS)
    errors = lag_errors(
        truth_times, f0_curve(signal, truth_times), grid, f0_hz, lags_ms / 1000.0
    )
    medians = np.array(
        [float(np.median(error)) if error.shape[0] else np.inf for error in errors]
    )
    best = int(np.argmin(medians))
    return Alignment(
        lag_ms=float(lags_ms[best]),
        cents=float(medians[best]),
        cents_at_zero=float(medians[int(np.argmin(np.abs(lags_ms)))]),
        voiced_share=float(np.mean(f0_hz[scored] > 0.0)),
    )


def format_report(results: dict[str, list[Alignment]]) -> str:
    header = " | ".join(signal.name for signal in SIGNALS)
    lines = [
        "Lag of each model against the synthetic truth, the median error at that "
        "lag and the voiced share of the scored frames. A model is aligned when "
        f"every lag stays within {ALIGNED_MS} ms, half a grid step.",
        "",
        f"| model | {header} | aligned |",
        "| --- | " + " | ".join("---:" for _ in SIGNALS) + " | --- |",
    ]
    for model, alignments in results.items():
        cells = " | ".join(
            f"{entry.lag_ms:+.1f} ms, {entry.cents:.1f} c, {entry.voiced_share:.0%}"
            for entry in alignments
        )
        aligned = all(abs(entry.lag_ms) <= ALIGNED_MS for entry in alignments)
        lines.append(f"| {model} | {cells} | {'yes' if aligned else 'no'} |")
    return "\n".join(lines) + "\n"


def main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = ModelContext(models_path=Path(args.models_path), device=device)
    audios = [synthesize(signal) for signal in SIGNALS]
    results = {}
    for name in args.models:
        model = ZOO_MODELS[name].create(context)
        results[name] = [
            align_model(model.estimate, signal, audio)
            for signal, audio in zip(SIGNALS, audios, strict=True)
        ]
    report = format_report(results)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
    sys.__stdout__.buffer.write(report.encode("utf-8"))
    sys.__stdout__.flush()
