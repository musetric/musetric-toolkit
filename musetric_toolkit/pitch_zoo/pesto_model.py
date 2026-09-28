import numpy as np
import pesto
import torch

from musetric_toolkit.pitch_zoo.estimate import SAMPLE_RATE, ModelContext, PitchEstimate

STEP_MS = 5.0
CONFIDENCE_THRESHOLD = 0.5


class PestoModel:
    def __init__(self, context: ModelContext) -> None:
        self.device = context.device

    @torch.no_grad()
    def estimate(self, audio: np.ndarray) -> PitchEstimate:
        timesteps, pitch, confidence, _ = pesto.predict(
            torch.from_numpy(audio.astype(np.float32)).to(self.device),
            SAMPLE_RATE,
            step_size=STEP_MS,
        )
        f0_hz = pitch.cpu().numpy().astype(np.float64)
        confidence = confidence.cpu().numpy().astype(np.float64)
        return PitchEstimate(
            times=timesteps.cpu().numpy().astype(np.float64) / 1000.0,
            f0_hz=np.where(confidence >= CONFIDENCE_THRESHOLD, f0_hz, 0.0),
            confidence=np.clip(confidence, 0.0, 1.0),
        )
