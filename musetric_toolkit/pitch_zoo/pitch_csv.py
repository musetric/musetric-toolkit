from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PitchTrack:
    times: np.ndarray
    f0_hz: np.ndarray
    confidence: np.ndarray
    trusted: np.ndarray


def read_pitch_csv(path: Path) -> PitchTrack:
    with path.open(encoding="utf-8") as csv_file:
        header = csv_file.readline().strip().split(",")
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    columns = {name: rows[:, index] for index, name in enumerate(header)}
    f0_hz = columns["f0_hz"]
    return PitchTrack(
        times=columns["time_s"],
        f0_hz=f0_hz,
        confidence=columns.get("confidence", (f0_hz > 0.0).astype(np.float64)),
        trusted=(
            columns["trusted"] == 1.0
            if "trusted" in columns
            else np.ones(f0_hz.shape[0], dtype=bool)
        ),
    )


def write_pitch_csv(path: Path, track: PitchTrack, with_trusted: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "time_s,f0_hz,confidence" + (",trusted" if with_trusted else "")
    with path.open("w", encoding="utf-8", newline="\n") as csv_file:
        csv_file.write(header + "\n")
        for index in range(track.times.shape[0]):
            row = (
                f"{track.times[index]:.4f},{track.f0_hz[index]:.3f},"
                f"{track.confidence[index]:.4f}"
            )
            if with_trusted:
                row += f",{int(track.trusted[index])}"
            csv_file.write(row + "\n")
