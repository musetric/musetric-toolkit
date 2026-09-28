import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musetric_toolkit.pitch_zoo.line import runs_of
from musetric_toolkit.pitch_zoo.pitch_csv import read_pitch_csv
from musetric_toolkit.pitch_zoo.score import list_tracks


@dataclass(frozen=True)
class Dispute:
    track: str
    from_seconds: float
    to_seconds: float
    reference_hz: float
    model_hz: dict[str, float]


def _median_voiced(values: np.ndarray) -> float:
    voiced = values[values > 0.0]
    return float(np.median(voiced)) if voiced.shape[0] else 0.0


def find_disputes(track: Path, name: str, models: list[str], min_ms: float) -> list:
    reference = read_pitch_csv(track / f"{name}.csv")
    parts = {model: read_pitch_csv(track / f"{model}.csv") for model in models}
    hop_ms = float(reference.times[1] - reference.times[0]) * 1000.0
    disputes = []
    for start, end in runs_of((reference.f0_hz > 0.0) & ~reference.trusted):
        if (end - start) * hop_ms < min_ms:
            continue
        disputes.append(
            Dispute(
                track=track.name,
                from_seconds=float(reference.times[start]),
                to_seconds=float(reference.times[end - 1]) + hop_ms / 1000.0,
                reference_hz=_median_voiced(reference.f0_hz[start:end]),
                model_hz={
                    model: _median_voiced(part.f0_hz[start:end])
                    for model, part in parts.items()
                },
            )
        )
    return disputes


def format_disputes(disputes: list[Dispute], models: list[str]) -> str:
    header = " | ".join(models)
    lines = [
        f"| track | from s | to s | reference Hz | {header} |",
        "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in models) + " |",
    ]
    for dispute in disputes:
        values = " | ".join(
            f"{dispute.model_hz[model]:.1f}" if dispute.model_hz[model] else "-"
            for model in models
        )
        lines.append(
            f"| {dispute.track} | {dispute.from_seconds:.3f} | "
            f"{dispute.to_seconds:.3f} | {dispute.reference_hz:.1f} | {values} |"
        )
    return "\n".join(lines) + "\n"


def main(args) -> None:
    tracks_dir = Path(args.tracks_dir)
    disputes = [
        dispute
        for track in list_tracks(tracks_dir, args.name)
        for dispute in find_disputes(track, args.name, args.models, args.min_ms)
    ]
    disputes.sort(key=lambda dispute: dispute.from_seconds - dispute.to_seconds)
    report = format_disputes(disputes[: args.count], args.models)
    (tracks_dir / f"{args.name}.disputes.md").write_text(report, encoding="utf-8")
    sys.__stdout__.buffer.write(report.encode("utf-8"))
    sys.__stdout__.flush()
