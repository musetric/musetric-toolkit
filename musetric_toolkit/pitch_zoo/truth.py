import hashlib
import io
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.common.model_files import ensure_model_file
from musetric_toolkit.pitch_zoo.estimate import SAMPLE_RATE, PitchEstimate, to_grid
from musetric_toolkit.pitch_zoo.pitch_csv import PitchTrack, write_pitch_csv
from musetric_toolkit.pitch_zoo.remote_zip import open_remote_zip
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.system_info import ensure_ffmpeg

VOCADITO_URL = "https://zenodo.org/records/5578807/files/vocadito.zip?download=1"
VOCADITO_MD5 = "dea40fd18f14d899643c4ba221b33a46"
DCS_URL = (
    "https://zenodo.org/records/4618287/files/DagstuhlChoirSet_V1.2.3.zip?download=1"
)
DCS_ROOT = "DagstuhlChoirSet_V1.2.3/"
DCS_MIC = "DYN"
PTDB_URL = "https://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH%20DATA/"
PTDB_SPEAKERS = tuple(f"F{index:02d}" for index in range(1, 11)) + tuple(
    f"M{index:02d}" for index in range(1, 11)
)
PTDB_SI_COUNT = 3
PTDB_HOP_SECONDS = 0.01
PTDB_FIRST_FRAME_SECONDS = 0.0205


@dataclass(frozen=True)
class Annotation:
    times: np.ndarray
    f0_hz: np.ndarray


@dataclass(frozen=True)
class TruthTrack:
    name: str
    audio_path: Path
    annotation: Annotation


def _time_f0(text: str) -> Annotation:
    rows = np.loadtxt(io.StringIO(text), delimiter=",", ndmin=2)
    return Annotation(times=rows[:, 0], f0_hz=np.maximum(rows[:, 1], 0.0))


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_vocadito(set_dir: Path, downloads: Path) -> list[TruthTrack]:
    archive_path = downloads / "vocadito.zip"
    ensure_model_file(VOCADITO_URL, archive_path, "vocadito")
    if _md5(archive_path) != VOCADITO_MD5:
        raise RuntimeError(f"{archive_path} does not match the Zenodo checksum")
    tracks = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            match = re.fullmatch(r"Audio/(vocadito_\d+)\.wav", member)
            if not match:
                continue
            name = match.group(1)
            audio_path = set_dir / "audio" / f"{name}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(archive.read(member))
            f0_member = f"Annotations/F0/{name}_f0.csv"
            annotation = _time_f0(archive.read(f0_member).decode("utf-8"))
            tracks.append(TruthTrack(name, audio_path, annotation))
    return tracks


def fetch_dcs(set_dir: Path, downloads: Path) -> list[TruthTrack]:
    archive = open_remote_zip(DCS_URL)
    manual = f"{DCS_ROOT}annotations_csv_F0_manual/"
    tracks = []
    for member in sorted(archive.namelist()):
        match = re.fullmatch(re.escape(manual) + r"(DCS_\w+)_LRX\.csv", member)
        if not match:
            continue
        name = f"{match.group(1)}_{DCS_MIC}"
        audio_path = set_dir / "audio" / f"{name}.wav"
        if not audio_path.exists():
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_member = f"{DCS_ROOT}audio_wav_22050_mono/{name}.wav"
            audio_path.write_bytes(archive.read(audio_member))
        annotation = _time_f0(archive.read(member).decode("utf-8"))
        tracks.append(TruthTrack(name, audio_path, annotation))
    return tracks


def _ptdb_listing(url: str) -> list[str]:
    with urlopen(url) as response:  # noqa: S310
        page = response.read().decode("utf-8")
    return re.findall(r'href="([^"?/]+)"', page)


def fetch_ptdb(set_dir: Path, downloads: Path) -> list[TruthTrack]:
    tracks = []
    for speaker in PTDB_SPEAKERS:
        sex = "FEMALE" if speaker.startswith("F") else "MALE"
        ref_url = f"{PTDB_URL}{sex}/REF/{speaker}/"
        sentences = [
            name[len(f"ref_{speaker}_") : -len(".f0")]
            for name in _ptdb_listing(ref_url)
            if name.endswith(".f0")
        ]
        numbered = sorted(
            (sentence for sentence in sentences if sentence.startswith("si")),
            key=lambda sentence: int(sentence[2:]),
        )
        for sentence in ["sa1", "sa2", *numbered[:PTDB_SI_COUNT]]:
            name = f"{speaker}_{sentence}"
            audio_path = set_dir / "audio" / f"{name}.wav"
            ensure_model_file(
                f"{PTDB_URL}{sex}/MIC/{speaker}/mic_{name}.wav", audio_path, name
            )
            ref_path = downloads / "ptdb" / f"ref_{name}.f0"
            ensure_model_file(f"{ref_url}ref_{name}.f0", ref_path, name)
            rows = np.loadtxt(ref_path, ndmin=2)
            annotation = Annotation(
                times=PTDB_FIRST_FRAME_SECONDS
                + np.arange(rows.shape[0]) * PTDB_HOP_SECONDS,
                f0_hz=np.where(rows[:, 1] > 0.0, rows[:, 0], 0.0),
            )
            tracks.append(TruthTrack(name, audio_path, annotation))
    return tracks


@dataclass(frozen=True)
class TruthSet:
    name: str
    license: str
    source: str
    fetch: Callable[[Path, Path], list[TruthTrack]]


TRUTH_SETS = {
    truth_set.name: truth_set
    for truth_set in (
        TruthSet(
            "vocadito",
            "CC BY 4.0",
            "https://zenodo.org/records/5578807",
            fetch_vocadito,
        ),
        TruthSet(
            "dcs",
            "CC BY 4.0",
            "https://zenodo.org/records/4618287",
            fetch_dcs,
        ),
        TruthSet(
            "ptdb",
            "ODbL 1.0 (database), DbCL 1.0 (contents)",
            "https://www.spsc.tugraz.at/databases-and-tools/"
            "ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html",
            fetch_ptdb,
        ),
    )
}


def write_truth(track_dir: Path, track: TruthTrack, hop_samples: int) -> None:
    audio = read_audio_file(str(track.audio_path), SAMPLE_RATE, 1)[0]
    times = np.arange(audio.shape[0] // hop_samples) * (hop_samples / SAMPLE_RATE)
    annotation = track.annotation
    voiced = (annotation.f0_hz > 0.0).astype(np.float64)
    grid = to_grid(PitchEstimate(annotation.times, annotation.f0_hz, voiced), times)
    inside = (times >= annotation.times[0]) & (times <= annotation.times[-1])
    f0_hz = np.where(inside, grid.f0_hz, 0.0)
    truth = PitchTrack(
        times=times,
        f0_hz=f0_hz,
        confidence=(f0_hz > 0.0).astype(np.float64),
        trusted=inside,
    )
    write_pitch_csv(track_dir / "truth.csv", truth, with_trusted=True)


def main(args) -> None:
    ensure_ffmpeg()
    data_dir = Path(args.data_dir)
    hop_samples = max(1, round(SAMPLE_RATE * args.hop_ms / 1000.0))
    for set_name in args.sets:
        truth_set = TRUTH_SETS[set_name]
        set_dir = data_dir / truth_set.name
        tracks = truth_set.fetch(set_dir, data_dir / "downloads")
        for index, track in enumerate(tracks):
            write_truth(set_dir / "tracks" / track.name, track, hop_samples)
            send_message(
                {
                    "type": "progress",
                    "set": truth_set.name,
                    "progress": (index + 1) / len(tracks),
                }
            )
