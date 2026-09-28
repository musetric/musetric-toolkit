import json
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.common.paths import list_audio_files
from musetric_toolkit.pitch_zoo.estimate import SAMPLE_RATE, ModelContext, to_grid
from musetric_toolkit.pitch_zoo.pitch_csv import PitchTrack, write_pitch_csv
from musetric_toolkit.pitch_zoo.registry import ZOO_MODELS, ZooModel
from musetric_toolkit.separate_audio.ffmpeg.read import read_audio_file
from musetric_toolkit.separate_audio.system_info import ensure_ffmpeg

MANIFEST_NAME = "zoo.json"


def write_manifest(path: Path, models: list[ZooModel], hop_ms: float) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for model in models:
        manifest[model.name] = {
            "package": model.package,
            "version": metadata.version(model.package),
            "license": model.license,
            "source": model.source,
            "hop_ms": hop_ms,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(args) -> None:
    ensure_ffmpeg()
    models = [ZOO_MODELS[name] for name in args.models]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = ModelContext(models_path=Path(args.models_path), device=device)
    estimators = [model.create(context) for model in models]
    hop_samples = max(1, round(SAMPLE_RATE * args.hop_ms / 1000.0))
    out_dir = Path(args.out_dir)
    audio_paths = list_audio_files(Path(args.audio_path))
    send_message({"type": "progress", "progress": 0.0})

    for index, audio_path in enumerate(audio_paths):
        audio = read_audio_file(str(audio_path), SAMPLE_RATE, 1)[0]
        frames = audio.shape[0] // hop_samples
        times = np.arange(frames) * (hop_samples / SAMPLE_RATE)
        for model, estimator in zip(models, estimators, strict=True):
            estimate = to_grid(estimator.estimate(audio), times)
            track = PitchTrack(
                times=times,
                f0_hz=estimate.f0_hz,
                confidence=estimate.confidence,
                trusted=estimate.f0_hz > 0.0,
            )
            path = out_dir / audio_path.stem / f"{model.name}.csv"
            write_pitch_csv(path, track, with_trusted=False)
        send_message({"type": "progress", "progress": (index + 1) / len(audio_paths)})

    write_manifest(out_dir / MANIFEST_NAME, models, args.hop_ms)
