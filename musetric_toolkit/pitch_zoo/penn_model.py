from pathlib import Path

import numpy as np
import penn
import torch

from musetric_toolkit.common.model_files import ensure_model_file
from musetric_toolkit.pitch_zoo.estimate import (
    FMAX_HZ,
    FMIN_HZ,
    SAMPLE_RATE,
    ModelContext,
    PitchEstimate,
)

CHECKPOINT_URL = (
    "https://huggingface.co/maxrmorrison/fcnf0-plus-plus/resolve/"
    "74911e26f43ad38790a42592e77f9d8be0a5dd1c/fcnf0%2B%2B.pt"
)
CHECKPOINT_REL_PATH = Path("penn") / "fcnf0++.pt"
HOP_SECONDS = 0.005
PERIODICITY_THRESHOLD = 0.065
BATCH_FRAMES = 2048


class PennModel:
    def __init__(self, context: ModelContext) -> None:
        self.checkpoint_path = context.models_path / CHECKPOINT_REL_PATH
        ensure_model_file(CHECKPOINT_URL, self.checkpoint_path, "PENN checkpoint")
        self.gpu = (
            (context.device.index or 0) if context.device.type == "cuda" else None
        )

    def estimate(self, audio: np.ndarray) -> PitchEstimate:
        pitch, periodicity = penn.from_audio(
            torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),
            SAMPLE_RATE,
            hopsize=HOP_SECONDS,
            fmin=FMIN_HZ,
            fmax=FMAX_HZ,
            checkpoint=self.checkpoint_path,
            batch_size=BATCH_FRAMES,
            center="zero",
            gpu=self.gpu,
        )
        f0_hz = pitch[0].cpu().numpy().astype(np.float64)
        confidence = periodicity[0].cpu().numpy().astype(np.float64)
        return PitchEstimate(
            times=np.arange(f0_hz.shape[0]) * HOP_SECONDS,
            f0_hz=np.where(confidence >= PERIODICITY_THRESHOLD, f0_hz, 0.0),
            confidence=np.clip(confidence, 0.0, 1.0),
        )
