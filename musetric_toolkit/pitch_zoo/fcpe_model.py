import numpy as np
import torch
from torchfcpe import spawn_bundled_infer_model

from musetric_toolkit.pitch_zoo.estimate import (
    FMAX_HZ,
    FMIN_HZ,
    SAMPLE_RATE,
    ModelContext,
    PitchEstimate,
)

THRESHOLD = 0.006


class FcpeModel:
    def __init__(self, context: ModelContext) -> None:
        self.device = context.device
        self.infer_model = spawn_bundled_infer_model(device=str(context.device))
        self.hop_seconds = (
            self.infer_model.get_hop_size() / self.infer_model.get_model_sr()
        )

    @torch.no_grad()
    def estimate(self, audio: np.ndarray) -> PitchEstimate:
        signal = torch.from_numpy(audio.astype(np.float32))[None, :, None]
        mel = self.infer_model.wav2mel(signal.to(self.device), SAMPLE_RATE)
        latent = self.infer_model.model.forward(mel)
        cents = self.infer_model.model.latent2cents_local_decoder(latent, mask=False)
        f0_hz = self.infer_model.model.cent_to_f0(cents)[0, :, 0].cpu().numpy()
        confidence = latent.max(dim=-1).values[0].cpu().numpy().astype(np.float64)
        voiced = (confidence > THRESHOLD) & (f0_hz >= FMIN_HZ) & (f0_hz <= FMAX_HZ)
        return PitchEstimate(
            times=np.arange(f0_hz.shape[0]) * self.hop_seconds,
            f0_hz=np.where(voiced, f0_hz.astype(np.float64), 0.0),
            confidence=np.clip(confidence, 0.0, 1.0),
        )
