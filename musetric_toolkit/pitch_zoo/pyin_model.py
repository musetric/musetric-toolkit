import librosa
import numpy as np

from musetric_toolkit.pitch_zoo.estimate import (
    FMAX_HZ,
    FMIN_HZ,
    SAMPLE_RATE,
    ModelContext,
    PitchEstimate,
)

HOP_SAMPLES = 160
FRAME_LENGTH = 1024


class PyinModel:
    def __init__(self, context: ModelContext) -> None:
        pass

    def estimate(self, audio: np.ndarray) -> PitchEstimate:
        f0_hz, voiced, probability = librosa.pyin(
            audio,
            fmin=FMIN_HZ,
            fmax=FMAX_HZ,
            sr=SAMPLE_RATE,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_SAMPLES,
        )
        return PitchEstimate(
            times=np.arange(f0_hz.shape[0]) * (HOP_SAMPLES / SAMPLE_RATE),
            f0_hz=np.where(voiced, np.nan_to_num(f0_hz), 0.0),
            confidence=np.clip(np.nan_to_num(probability), 0.0, 1.0),
        )
