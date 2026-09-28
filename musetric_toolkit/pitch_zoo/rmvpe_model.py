import numpy as np

from musetric_toolkit.common import envs
from musetric_toolkit.common.model_files import ensure_model_file
from musetric_toolkit.pitch_audio.tracker import SAMPLE_RATE, load_tracker, track
from musetric_toolkit.pitch_audio.trusted import energy_gate
from musetric_toolkit.pitch_zoo.estimate import ModelContext, PitchEstimate

HOP_SAMPLES = 80


class RmvpeModel:
    def __init__(self, context: ModelContext) -> None:
        checkpoint_path = context.models_path / envs.rmvpe_checkpoint_rel_path
        ensure_model_file(
            envs.rmvpe_checkpoint_url, checkpoint_path, "RMVPE checkpoint"
        )
        self.tracker = load_tracker(checkpoint_path, HOP_SAMPLES)

    def estimate(self, audio: np.ndarray) -> PitchEstimate:
        result = track(self.tracker, audio)
        voiced = energy_gate(
            audio, SAMPLE_RATE, HOP_SAMPLES, result.f0_hz, result.confidence
        )
        return PitchEstimate(
            times=np.arange(result.f0_hz.shape[0]) * (HOP_SAMPLES / SAMPLE_RATE),
            f0_hz=np.where(voiced, result.f0_hz, 0.0),
            confidence=np.clip(result.confidence, 0.0, 1.0),
        )
