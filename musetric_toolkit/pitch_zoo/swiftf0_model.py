import numpy as np
from swift_f0 import SwiftF0

from musetric_toolkit.pitch_zoo.estimate import (
    FMAX_HZ,
    FMIN_HZ,
    SAMPLE_RATE,
    ModelContext,
    PitchEstimate,
)

VOICED_CONFIDENCE = 0.5


class SwiftF0Model:
    def __init__(self, context: ModelContext) -> None:
        self.detector = SwiftF0()

    def estimate(self, audio: np.ndarray) -> PitchEstimate:
        result = self.detector.detect(
            audio.astype(np.float32), SAMPLE_RATE, FMIN_HZ, FMAX_HZ
        )
        return PitchEstimate(
            times=result.timestamps,
            f0_hz=np.where(
                result.confidence >= VOICED_CONFIDENCE, result.pitch_hz, 0.0
            ),
            confidence=np.clip(result.confidence, 0.0, 1.0),
        )
