import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from musetric_toolkit.pitch_zoo.accuracy import (
    TOLERANCE_CENTS,
    PitchAccuracy,
    cents_between,
    compare_accuracy,
    ratio,
)
from musetric_toolkit.pitch_zoo.line import LineOverview, LineWindow, compare_line
from musetric_toolkit.pitch_zoo.pitch_csv import PitchTrack, read_pitch_csv


@dataclass(frozen=True)
class TrustScore:
    trusted_share: float
    trusted_rpa: float
    trusted_false_voicing: float


def score_trust(truth: PitchTrack, candidate: PitchTrack) -> TrustScore:
    index = np.searchsorted(candidate.times, truth.times, side="right") - 1
    held = np.maximum(index, 0)
    f0_hz = np.where(index >= 0, candidate.f0_hz[held], 0.0)
    trusted = (index >= 0) & candidate.trusted[held] & (f0_hz > 0.0)
    voice = truth.trusted & (truth.f0_hz > 0.0)
    on_voice = trusted & voice
    errors = np.abs(cents_between(f0_hz[on_voice], truth.f0_hz[on_voice]))
    return TrustScore(
        trusted_share=ratio(np.count_nonzero(on_voice), np.count_nonzero(voice)),
        trusted_rpa=ratio(
            np.count_nonzero(errors <= TOLERANCE_CENTS), np.count_nonzero(on_voice)
        ),
        trusted_false_voicing=ratio(
            np.count_nonzero(trusted & (truth.f0_hz <= 0.0)), np.count_nonzero(trusted)
        ),
    )


@dataclass(frozen=True)
class TrackScore:
    accuracy: PitchAccuracy
    line: LineOverview
    trust: TrustScore
    worst: list[LineWindow]


def score_track(
    truth: PitchTrack, candidate: PitchTrack, worst_count: int
) -> TrackScore:
    line = compare_line(truth, candidate, worst_count)
    return TrackScore(
        accuracy=compare_accuracy(truth, candidate),
        line=line.overview,
        trust=score_trust(truth, candidate),
        worst=line.worst,
    )


@dataclass(frozen=True)
class Column:
    title: str
    group: str
    key: str
    format: Callable[[float], str]


def _fixed(digits: int) -> Callable[[float], str]:
    return lambda value: f"{value:.{digits}f}"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


TABLES = (
    (
        "Accuracy",
        (
            Column("rpa", "accuracy", "rpa", _fixed(3)),
            Column("rca", "accuracy", "rca", _fixed(3)),
            Column("recall", "accuracy", "voicing_recall", _fixed(3)),
            Column("false alarm", "accuracy", "false_alarm", _fixed(3)),
            Column("bias ¢", "accuracy", "bias_cents", _fixed(1)),
            Column("spread ¢", "accuracy", "spread_cents", _fixed(1)),
            Column("octave", "accuracy", "octave_rate", _fixed(3)),
            Column("gap flip", "accuracy", "gap_flip_rate", _fixed(3)),
            Column("note break", "accuracy", "note_break_rate", _fixed(3)),
        ),
    ),
    (
        "Line: share of the singing time",
        (
            Column("clean", "line", "clean_share", _percent),
            Column("missing", "line", "missing_share", _percent),
            Column("wrong", "line", "wrong_share", _percent),
            Column("jerky", "line", "jerky_share", _percent),
            Column("rattling", "line", "rattling_share", _percent),
            Column("rattle ¢", "line", "rattle_cents", _fixed(1)),
            Column("rattle p90 ¢", "line", "rattle_p90_cents", _fixed(1)),
        ),
    ),
    (
        "Line: events per minute of singing",
        (
            Column("steps", "line", "steps_per_minute", _fixed(1)),
            Column("leaps", "line", "leaps_per_minute", _fixed(1)),
            Column("holes", "line", "holes_per_minute", _fixed(1)),
            Column("long holes", "line", "long_holes_per_minute", _fixed(1)),
            Column("specks", "line", "specks_per_minute", _fixed(1)),
            Column("islands", "line", "islands_per_minute", _fixed(1)),
            Column("tails", "line", "tails_per_minute", _fixed(1)),
            Column("onset ms", "line", "onset_delay_ms", _fixed(0)),
            Column("onset p90 ms", "line", "onset_delay_p90_ms", _fixed(0)),
            Column("missed onsets", "line", "missed_onset_share", _percent),
        ),
    ),
    (
        "Trusted frames",
        (
            Column("trusted share", "trust", "trusted_share", _percent),
            Column("trusted rpa", "trust", "trusted_rpa", _fixed(3)),
            Column(
                "trusted false voicing", "trust", "trusted_false_voicing", _fixed(3)
            ),
        ),
    ),
)


def mean_scores(scores: list[dict]) -> dict:
    return {
        group: {
            key: float(np.mean([score[group][key] for score in scores]))
            for key in scores[0][group]
        }
        for group in ("accuracy", "line", "trust")
    }


VOICING_KEYS = (
    "false_alarm",
    "islands_per_minute",
    "tails_per_minute",
    "trusted_false_voicing",
)


def format_report(
    truth_name: str, track_count: int, means: dict[str, dict], voicing: bool
) -> str:
    parts = [f"Mean of {track_count} tracks against `{truth_name}`."]
    if not voicing:
        parts.append("Voicing columns left out: the truth marks voice as unvoiced.")
    for title, all_columns in TABLES:
        columns = [
            column
            for column in all_columns
            if voicing or column.key not in VOICING_KEYS
        ]
        header = " | ".join(column.title for column in columns)
        rule = " | ".join("---:" for _ in columns)
        rows = [
            f"| {model} | "
            + " | ".join(
                column.format(values[column.group][column.key]) for column in columns
            )
            + " |"
            for model, values in means.items()
        ]
        parts.append(
            "\n".join(
                [f"### {title}", "", f"| model | {header} |", f"| --- | {rule} |"]
            )
            + "\n"
            + "\n".join(rows)
        )
    return "\n\n".join(parts) + "\n"


def list_tracks(tracks_dir: Path, truth_name: str) -> list[Path]:
    return sorted(
        path.parent for path in tracks_dir.glob(f"*/{truth_name}.csv") if path.is_file()
    )


def main(args) -> None:
    tracks_dir = Path(args.tracks_dir)
    tracks = list_tracks(tracks_dir, args.truth)
    per_model: dict[str, dict[str, dict]] = {model: {} for model in args.models}
    for track in tracks:
        truth = read_pitch_csv(track / f"{args.truth}.csv")
        for model in args.models:
            score = score_track(
                truth, read_pitch_csv(track / f"{model}.csv"), args.worst_count
            )
            per_model[model][track.name] = asdict(score)
    means = {
        model: mean_scores(list(scores.values())) for model, scores in per_model.items()
    }
    report = format_report(args.truth, len(tracks), means, args.voicing)
    out = Path(args.out) if args.out else tracks_dir / f"score-{args.truth}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(report, encoding="utf-8")
    result = {"truth": args.truth, "means": means, "tracks": per_model}
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    sys.__stdout__.buffer.write(report.encode("utf-8"))
    sys.__stdout__.flush()
