import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    MultipleLocator,
    NullFormatter,
)

from musetric_toolkit.pitch_zoo.accuracy import cents_between
from musetric_toolkit.pitch_zoo.line import compare_line, runs_of
from musetric_toolkit.pitch_zoo.pitch_csv import PitchTrack, read_pitch_csv
from musetric_toolkit.pitch_zoo.score import list_tracks
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.system_info import ensure_ffmpeg

SAMPLE_RATE = 48000
N_FFT = 16384
HOP = 192
BANDS = ((8192, 150.0), (4096, 600.0), (2048, 1600.0))
FMIN_HZ = 50.0
FMAX_HZ = 1600.0
DB_RANGE = 58.0
PAD_SECONDS = 0.2
SEGMENT_SPAN_CENTS = 720.0
PAD_SEMITONES = 1.0
MIN_SPAN_SEMITONES = 10.0
LABELED_ROWS = 24
WINDOW_SECONDS = 5.0
WINDOW_STEP_SECONDS = 1.0
DIFFER_CENTS = 50.0
NATURAL_NOTES = {0: "C", 2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 11: "B"}


def _midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def _hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


@dataclass(frozen=True)
class Window:
    track: str
    from_seconds: float
    to_seconds: float
    label: str

    @property
    def name(self) -> str:
        return f"{self.from_seconds:g}-{self.to_seconds:g}s"


def _voiced_in(track: PitchTrack, lo: float, hi: float) -> np.ndarray:
    inside = (track.times >= lo) & (track.times <= hi) & (track.f0_hz > 0.0)
    values = track.f0_hz[inside]
    return values[(values >= FMIN_HZ) & (values <= FMAX_HZ)]


def frequency_range(
    reference: PitchTrack, compare: PitchTrack, lo: float, hi: float
) -> tuple[float, float]:
    values = np.concatenate(
        [_voiced_in(reference, lo, hi), _voiced_in(compare, lo, hi)]
    )
    if values.shape[0] == 0:
        return 80.0, 700.0
    low = _midi(float(values.min())) - PAD_SEMITONES
    high = _midi(float(values.max())) + PAD_SEMITONES
    if high - low < MIN_SPAN_SEMITONES:
        center = (high + low) / 2.0
        low, high = center - MIN_SPAN_SEMITONES / 2.0, center + MIN_SPAN_SEMITONES / 2.0
    return max(FMIN_HZ, _hz(low)), min(FMAX_HZ, _hz(high))


def _time_axis(axes, lo: float, hi: float) -> None:
    axes.set_xlim(lo, hi)
    axes.xaxis.set_major_locator(MultipleLocator(0.5))
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes.xaxis.set_minor_locator(MultipleLocator(0.1))
    axes.grid(visible=True, which="major", axis="x", alpha=0.28)
    axes.grid(visible=True, which="minor", axis="x", alpha=0.08)


def _hz_axis(axes, hz_range: tuple[float, float]) -> None:
    low = math.floor(_midi(hz_range[0]))
    high = math.ceil(_midi(hz_range[1]))
    rows = list(range(low, high + 1))
    every = max(1, math.ceil((high - low) / LABELED_ROWS))
    axes.set_yscale("log")
    axes.set_ylim(*hz_range)
    axes.yaxis.set_major_locator(FixedLocator([_hz(row) for row in rows]))
    axes.set_yticklabels(
        [
            f"{round(_hz(row))}" if index % every == 0 else ""
            for index, row in enumerate(rows)
        ],
        fontsize=7,
    )
    axes.yaxis.set_minor_formatter(NullFormatter())
    axes.grid(visible=True, which="major", axis="y", alpha=0.18)


def draw_line(axes, track: PitchTrack, lo: float, hi: float, style: dict) -> None:
    inside = (track.times >= lo) & (track.times <= hi)
    times = track.times[inside]
    f0_hz = track.f0_hz[inside]
    both = (f0_hz[1:] > 0.0) & (f0_hz[:-1] > 0.0)
    span = np.full(both.shape[0], np.inf)
    span[both] = np.abs(cents_between(f0_hz[1:][both], f0_hz[:-1][both]))
    frames = style.pop("keep", np.ones(times.shape[0], dtype=bool))
    keep = frames[:-1] & frames[1:] & (span <= SEGMENT_SPAN_CENTS)
    starts = np.stack((times[:-1], f0_hz[:-1]), axis=1)[keep]
    ends = np.stack((times[1:], f0_hz[1:]), axis=1)[keep]
    axes.add_collection(LineCollection(np.stack((starts, ends), axis=1), **style))


@dataclass(frozen=True)
class Panel:
    title: str
    color: str
    with_trust: bool


def draw_panel(axes, track: PitchTrack, lo: float, hi: float, panel: Panel) -> None:
    inside = (track.times >= lo) & (track.times <= hi)
    trusted = track.trusted[inside]
    solid = {"colors": panel.color, "linewidths": 1.8, "alpha": 0.9}
    draw_line(axes, track, lo, hi, {**solid, "keep": trusted})
    draw_line(
        axes,
        track,
        lo,
        hi,
        {**solid, "linewidths": 0.9, "linestyles": ":", "alpha": 0.6, "keep": ~trusted},
    )
    axes.set_title(panel.title, loc="left", fontsize=10)
    if not panel.with_trust:
        return
    axes.plot([], [], color=panel.color, lw=1.8, label="trusted")
    axes.plot([], [], color=panel.color, lw=0.9, ls=":", label="voiced, not trusted")
    axes.legend(loc="upper right", fontsize=9, framealpha=0.92)


def spectrum_of(audio: np.ndarray, lo: float, hi: float) -> tuple:
    start = max(0.0, lo - PAD_SECONDS)
    segment = torch.from_numpy(
        audio[int(start * SAMPLE_RATE) : int((hi + PAD_SECONDS) * SAMPLE_RATE)]
    )
    frequencies = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    keep = (frequencies >= FMIN_HZ) & (frequencies <= FMAX_HZ)
    rows = None
    band_low = FMIN_HZ
    for length, band_high in BANDS:
        window = torch.hann_window(length)
        magnitude = (
            torch.stft(
                segment,
                n_fft=N_FFT,
                hop_length=HOP,
                win_length=length,
                window=window,
                center=True,
                return_complex=True,
            )
            .abs()
            .numpy()
        ) / (float(window.sum()) / 2.0)
        if rows is None:
            rows = np.zeros((int(keep.sum()), magnitude.shape[1]), dtype=np.float32)
        band = keep & (frequencies >= band_low) & (frequencies < band_high)
        rows[band[keep]] = magnitude[band]
        band_low = band_high
    times = start + np.arange(rows.shape[1]) * HOP / SAMPLE_RATE
    return times, frequencies[keep], 20.0 * np.log10(rows + 1e-8)


@dataclass(frozen=True)
class WindowView:
    window: Window
    reference: PitchTrack
    compare: PitchTrack
    hz_range: tuple[float, float]


def render_spectrum(
    audio: np.ndarray, view: WindowView, out_path: Path, dpi: int
) -> None:
    lo, hi = view.window.from_seconds, view.window.to_seconds
    reference, hz_range = view.reference, view.hz_range
    times, frequencies, decibels = spectrum_of(audio, lo, hi)
    figure = Figure(figsize=(max(14.0, (times[-1] - times[0]) * 3.2), 10), dpi=dpi)
    axes = figure.subplots()
    axes.pcolormesh(
        times,
        frequencies,
        decibels,
        vmin=decibels.max() - DB_RANGE,
        vmax=decibels.max(),
        cmap="magma",
        shading="gouraud",
    )
    _hz_axis(axes, hz_range)
    for row in range(math.floor(_midi(hz_range[0])), math.ceil(_midi(hz_range[1]))):
        if row % 12 in NATURAL_NOTES and hz_range[0] <= _hz(row) <= hz_range[1]:
            axes.axhline(_hz(row), color="cyan", lw=0.35, alpha=0.5)
            name = f"{NATURAL_NOTES[row % 12]}{row // 12 - 1}"
            axes.text(times[0] + 0.01, _hz(row), name, color="cyan", fontsize=8)
    inside = (reference.times >= times[0]) & (reference.times <= times[-1])
    draw_line(
        axes,
        reference,
        float(times[0]),
        float(times[-1]),
        {
            "colors": "deepskyblue",
            "linewidths": 1.2,
            "alpha": 0.9,
            "keep": reference.trusted[inside],
        },
    )
    axes.plot([], [], color="deepskyblue", lw=1.2, label="reference (trusted)")
    axes.axvline(lo, color="lime", lw=0.7, alpha=0.7)
    axes.axvline(hi, color="lime", lw=0.7, alpha=0.7)
    axes.legend(loc="upper right", fontsize=9, framealpha=0.9)
    _time_axis(axes, float(times[0]), float(times[-1]))
    axes.set_xlabel("time (s)")
    axes.set_ylabel("Hz (log)")
    figure.tight_layout()
    figure.savefig(out_path)


@dataclass(frozen=True)
class Panels:
    compare: Panel
    reference: Panel


def render_overlay(view: WindowView, panels: Panels, out_path: Path, dpi: int) -> None:
    lo, hi = view.window.from_seconds, view.window.to_seconds
    reference, compare, hz_range = view.reference, view.compare, view.hz_range
    figure = Figure(figsize=(18, 12), dpi=dpi)
    compare_axes, reference_axes, confidence_axes = figure.subplots(
        3, 1, sharex=True, gridspec_kw={"height_ratios": [2.2, 2.2, 1.0]}
    )
    draw_panel(compare_axes, compare, lo, hi, panels.compare)
    draw_panel(reference_axes, reference, lo, hi, panels.reference)
    for axes in (compare_axes, reference_axes):
        _hz_axis(axes, hz_range)
        axes.set_ylabel("F0 (Hz, log)")
        _time_axis(axes, lo, hi)
    inside = (reference.times >= lo) & (reference.times <= hi)
    confidence_axes.plot(
        reference.times[inside], reference.confidence[inside], color="red", lw=1.0
    )
    confidence_axes.set_ylim(0, 1.05)
    confidence_axes.set_ylabel("confidence")
    confidence_axes.set_xlabel("time (s)")
    confidence_axes.set_title("reference confidence", loc="left", fontsize=10)
    confidence_axes.grid(visible=True, axis="y", alpha=0.2)
    _time_axis(confidence_axes, lo, hi)
    figure.tight_layout()
    figure.savefig(out_path)


def _pick_apart(candidates: list[tuple[float, Window]], count: int) -> list[Window]:
    picked: list[Window] = []
    for _, window in sorted(candidates, key=lambda entry: -entry[0]):
        apart = all(
            window.track != other.track
            or window.to_seconds <= other.from_seconds
            or window.from_seconds >= other.to_seconds
            for other in picked
        )
        if apart:
            picked.append(window)
        if len(picked) == count:
            break
    return picked


def _centered(track: str, times: np.ndarray, start: int, end: int) -> tuple:
    middle = (times[start] + times[end - 1]) / 2.0
    duration = float(times[-1])
    lo = min(
        max(0.0, middle - WINDOW_SECONDS / 2.0), max(0.0, duration - WINDOW_SECONDS)
    )
    return track, round(lo, 2), round(lo + WINDOW_SECONDS, 2)


def select_worst(pairs: dict, count: int) -> list[Window]:
    candidates = []
    for track, (reference, compare) in pairs.items():
        for window in compare_line(reference, compare, count).worst:
            label = f"{window.bad_share * 100:.0f}% not clean"
            lo = round(window.from_seconds, 2)
            candidates.append(
                (
                    window.bad_share,
                    Window(track, lo, round(lo + WINDOW_SECONDS, 2), label),
                )
            )
    return _pick_apart(candidates, count)


def select_disputes(pairs: dict, count: int) -> list[Window]:
    candidates = []
    for track, (reference, _) in pairs.items():
        hop_ms = float(reference.times[1] - reference.times[0]) * 1000.0
        for start, end in runs_of((reference.f0_hz > 0.0) & ~reference.trusted):
            name, lo, hi = _centered(track, reference.times, start, end)
            label = f"{(end - start) * hop_ms:.0f} ms disputed"
            candidates.append(((end - start) * hop_ms, Window(name, lo, hi, label)))
    return _pick_apart(candidates, count)


def select_differ(pairs: dict, count: int) -> list[Window]:
    candidates = []
    for track, (reference, compare) in pairs.items():
        frames = min(reference.times.shape[0], compare.times.shape[0])
        new, old = reference.f0_hz[:frames], compare.f0_hz[:frames]
        both = compare.trusted[:frames] & (old > 0.0) & (new > 0.0)
        apart = np.zeros(frames, dtype=bool)
        apart[both] = np.abs(cents_between(new[both], old[both])) > DIFFER_CENTS
        hop = float(reference.times[1] - reference.times[0])
        size = round(WINDOW_SECONDS / hop)
        for start in range(0, frames - size + 1, round(WINDOW_STEP_SECONDS / hop)):
            frames_apart = int(np.count_nonzero(apart[start : start + size]))
            if frames_apart:
                lo = round(float(reference.times[start]), 2)
                label = f"{frames_apart * hop * 1000:.0f} ms apart"
                window = Window(track, lo, round(lo + WINDOW_SECONDS, 2), label)
                candidates.append((frames_apart, window))
    return _pick_apart(candidates, count)


SELECTIONS = {
    "worst": select_worst,
    "disputes": select_disputes,
    "differ": select_differ,
}


def _explicit(specs: list[str]) -> list[Window]:
    windows = []
    for spec in specs:
        track, span = spec.rsplit(":", 1)
        lo, hi = (float(value) for value in span.split("-"))
        windows.append(Window(track, lo, hi, "requested"))
    return windows


def _slice(source: Path, window: Window, target: Path) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    rows = [
        line
        for line in lines[1:]
        if window.from_seconds <= float(line.split(",", 1)[0]) <= window.to_seconds
    ]
    target.write_text("\n".join([lines[0], *rows]) + "\n", encoding="utf-8")


def _has_trust(path: Path) -> bool:
    with path.open(encoding="utf-8") as csv_file:
        return "trusted" in csv_file.readline().strip().split(",")


def main(args) -> None:
    ensure_ffmpeg()
    tracks_dir, audio_dir, out_dir = (
        Path(args.tracks_dir),
        Path(args.audio_dir),
        Path(args.out_dir),
    )
    tracks = [track.name for track in list_tracks(tracks_dir, args.reference)]
    if args.window:
        windows = _explicit(args.window)
    else:
        pairs = {
            track: (
                read_pitch_csv(tracks_dir / track / f"{args.reference}.csv"),
                read_pitch_csv(tracks_dir / track / f"{args.compare}.csv"),
            )
            for track in tracks
        }
        windows = SELECTIONS[args.select](pairs, args.count)
    panels = Panels(
        compare=Panel(
            title=args.compare_title or args.compare,
            color="C0",
            with_trust=_has_trust(tracks_dir / tracks[0] / f"{args.compare}.csv"),
        ),
        reference=Panel(
            title=args.reference_title or args.reference,
            color="red",
            with_trust=_has_trust(tracks_dir / tracks[0] / f"{args.reference}.csv"),
        ),
    )
    rows = ["| window | why |", "| --- | --- |"]
    for window in windows:
        reference_path = tracks_dir / window.track / f"{args.reference}.csv"
        compare_path = tracks_dir / window.track / f"{args.compare}.csv"
        reference = read_pitch_csv(reference_path)
        compare = read_pitch_csv(compare_path)
        audio_path = next(audio_dir.glob(f"{window.track}.*"))
        audio = read_audio_file(str(audio_path), SAMPLE_RATE, 1)[0]
        window_dir = out_dir / window.track / window.name
        (window_dir / "csv").mkdir(parents=True, exist_ok=True)
        _slice(reference_path, window, window_dir / "csv" / f"{args.reference}.csv")
        _slice(compare_path, window, window_dir / "csv" / f"{args.compare}.csv")
        view = WindowView(
            window=window,
            reference=reference,
            compare=compare,
            hz_range=frequency_range(
                reference, compare, window.from_seconds, window.to_seconds
            ),
        )
        render_spectrum(audio, view, window_dir / "spectrum.png", args.dpi)
        render_overlay(view, panels, window_dir / "overlay.png", args.dpi)
        rows.append(f"| `{window.track}/{window.name}` | {window.label} |")
    (out_dir / "windows.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
