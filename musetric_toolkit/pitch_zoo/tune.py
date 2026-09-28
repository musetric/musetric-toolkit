import itertools
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from musetric_toolkit.pitch_audio.ensemble import EnsembleParams, ModelWeights, combine
from musetric_toolkit.pitch_zoo.pitch_csv import PitchTrack, read_pitch_csv
from musetric_toolkit.pitch_zoo.score import list_tracks, score_track

METRICS = (
    ("line", "clean_share"),
    ("line", "jerky_share"),
    ("line", "rattle_cents"),
    ("trust", "trusted_rpa"),
    ("trust", "trusted_share"),
    ("accuracy", "false_alarm"),
)


@dataclass(frozen=True)
class Candidate:
    label: str
    weights: ModelWeights | None
    params: EnsembleParams | None
    model: str | None = None


@dataclass(frozen=True)
class SetTrack:
    truth: PitchTrack
    parts: list[PitchTrack]


def _weights_label(values: list[float]) -> str:
    return ":".join(f"{value:g}" for value in values)


def build_candidates(grid: dict, models: list[str]) -> list[Candidate]:
    ones = [1.0] * len(models)
    overrides = grid.get("params", {})
    keys = sorted(overrides)
    candidates = [Candidate(model, None, None, model) for model in models]
    for voicing, pitch, anchors, values in itertools.product(
        grid.get("voicing_weights", [ones]),
        grid.get("pitch_weights", [ones]),
        grid.get("anchors", [[]]),
        itertools.product(*(overrides[key] for key in keys)),
    ):
        changed = dict(zip(keys, values, strict=True))
        label = f"voicing {_weights_label(voicing)}, pitch {_weights_label(pitch)}"
        if anchors:
            label += f", anchors {_weights_label(anchors)}"
        label += "".join(f", {key} {value:g}" for key, value in changed.items())
        weights = ModelWeights(
            voicing=np.array(voicing, dtype=np.float64),
            pitch=np.array(pitch, dtype=np.float64),
            anchors=tuple(anchors),
        )
        candidates.append(
            Candidate(label, weights, replace(EnsembleParams(), **changed))
        )
    return candidates


def load_set(tracks_dir: Path, truth: str, models: list[str]) -> list[SetTrack]:
    return [
        SetTrack(
            truth=read_pitch_csv(track / f"{truth}.csv"),
            parts=[read_pitch_csv(track / f"{model}.csv") for model in models],
        )
        for track in list_tracks(tracks_dir, truth)
    ]


def candidate_track(candidate: Candidate, models: list[str], parts) -> PitchTrack:
    if candidate.model is not None:
        return parts[models.index(candidate.model)]
    frames = min(part.times.shape[0] for part in parts)
    f0_hz = np.stack([part.f0_hz[:frames] for part in parts])
    combined = combine(f0_hz, candidate.weights, candidate.params)
    return PitchTrack(
        times=parts[0].times[:frames],
        f0_hz=combined.f0_hz,
        confidence=combined.confidence,
        trusted=combined.trusted,
    )


def evaluate(
    candidate: Candidate, models: list[str], tracks: list[SetTrack]
) -> dict[str, float]:
    scores = [
        asdict(
            score_track(entry.truth, candidate_track(candidate, models, entry.parts), 0)
        )
        for entry in tracks
    ]
    return {
        key: float(np.mean([score[group][key] for score in scores]))
        for group, key in METRICS
    }


def set_name(tracks_dir: Path) -> str:
    return tracks_dir.parent.name if tracks_dir.name == "tracks" else tracks_dir.name


TABLES = (
    ("Clean line, share of the singing", "clean_share", "{:.1%}"),
    ("Trusted frames within 50 cents of the truth", "trusted_rpa", "{:.3f}"),
    ("Share of the voiced truth trusted", "trusted_share", "{:.1%}"),
    ("False alarm, share of the unvoiced truth voiced", "false_alarm", "{:.3f}"),
)


def format_report(results: dict[str, dict[str, dict[str, float]]]) -> str:
    sets = list(next(iter(results.values())))
    ranked = sorted(
        results.items(),
        key=lambda item: -np.mean([item[1][name]["clean_share"] for name in sets]),
    )
    parts = ["Candidates in the order of their mean clean share over the sets."]
    for title, key, number in TABLES:
        lines = [
            f"### {title}",
            "",
            f"| candidate | {' | '.join(sets)} | mean |",
            "| --- | " + " | ".join("---:" for _ in [*sets, "mean"]) + " |",
        ]
        for label, by_set in ranked:
            values = [by_set[name][key] for name in sets]
            cells = " | ".join(number.format(value) for value in values)
            mean = number.format(float(np.mean(values)))
            lines.append(f"| {label} | {cells} | {mean} |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"


def main(args) -> None:
    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    candidates = build_candidates(grid, args.models)
    sets = {
        set_name(Path(tracks_dir)): load_set(Path(tracks_dir), args.truth, args.models)
        for tracks_dir in args.tracks_dirs
    }
    results = {
        candidate.label: {
            name: evaluate(candidate, args.models, tracks)
            for name, tracks in sets.items()
        }
        for candidate in candidates
    }
    report = format_report(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
    sys.__stdout__.buffer.write(report.encode("utf-8"))
    sys.__stdout__.flush()
