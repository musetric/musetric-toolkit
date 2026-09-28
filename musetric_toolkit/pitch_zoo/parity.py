import json
import sys
from dataclasses import asdict
from pathlib import Path

from musetric_toolkit.pitch_zoo.accuracy import compare_accuracy
from musetric_toolkit.pitch_zoo.line import compare_line
from musetric_toolkit.pitch_zoo.pitch_csv import read_pitch_csv

TOLERANCE = 1e-9


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.title() for part in rest)


def main(args) -> None:
    tracks_dir = Path(args.tracks_dir)
    reports = json.loads(Path(args.bench_report).read_text(encoding="utf-8"))
    mismatches = []
    for report in reports:
        track = tracks_dir / report["name"]
        truth = read_pitch_csv(track / "reference.csv")
        candidate = read_pitch_csv(track / f"{args.tag}.csv")
        line = compare_line(truth, candidate, len(report["worst"]))
        groups = (
            ("accuracy", asdict(compare_accuracy(truth, candidate))),
            ("line", asdict(line.overview)),
        )
        for group, values in groups:
            for key, value in values.items():
                expected = report[group][_camel(key)]
                if abs(value - expected) > TOLERANCE * max(1.0, abs(expected)):
                    mismatches.append(f"{report['name']} {group}.{key}")
        windows = [(window.from_seconds, window.bad_share) for window in line.worst]
        expected_windows = [
            (window["fromSeconds"], window["badShare"]) for window in report["worst"]
        ]
        if len(windows) != len(expected_windows) or any(
            ours[0] != theirs[0] or abs(ours[1] - theirs[1]) > TOLERANCE
            for ours, theirs in zip(windows, expected_windows, strict=False)
        ):
            mismatches.append(f"{report['name']} worst windows")
    summary = f"{len(reports)} tracks, {len(mismatches)} mismatches\n"
    sys.__stdout__.write(summary + "".join(f"{entry}\n" for entry in mismatches))
    if mismatches:
        raise RuntimeError("the Python scores differ from the bench report")
