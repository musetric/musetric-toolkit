import numpy as np
import torch
import torchcrepe

from musetric_toolkit.pitch_audio.tracker import (
    CENTS_OFFSET,
    CENTS_STEP,
    N_BINS,
    create_salience_decoder,
    decode_salience,
)
from musetric_toolkit.pitch_zoo.estimate import (
    FMAX_HZ,
    FMIN_HZ,
    SAMPLE_RATE,
    ModelContext,
    PitchEstimate,
)

CAPACITY = "full"
HOP_SAMPLES = 80
PERIODICITY_THRESHOLD = 0.21
SILENCE_DB = -60.0
BATCH_SIZE = 512


def _band() -> np.ndarray:
    cents = CENTS_STEP * np.arange(N_BINS) + CENTS_OFFSET
    hz = 10.0 * np.power(2.0, cents / 1200.0)
    return (hz >= FMIN_HZ) & (hz <= FMAX_HZ)


class CrepeModel:
    def __init__(self, context: ModelContext) -> None:
        self.device = context.device
        self.decoder = create_salience_decoder()
        self.band = _band()

    @torch.no_grad()
    def estimate(self, audio: np.ndarray) -> PitchEstimate:
        signal = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        parts = [
            torchcrepe.infer(frames, CAPACITY, self.device).cpu()
            for frames in torchcrepe.preprocess(
                signal, SAMPLE_RATE, HOP_SAMPLES, BATCH_SIZE, self.device
            )
        ]
        salience = torch.cat(parts).numpy().astype(np.float64)
        salience[:, ~self.band] = 0.0
        result = decode_salience(self.decoder, salience)
        periodicity = torchcrepe.threshold.Silence(SILENCE_DB)(
            torch.from_numpy(result.confidence[None, :]),
            signal,
            SAMPLE_RATE,
            HOP_SAMPLES,
        )[0].numpy()
        return PitchEstimate(
            times=np.arange(result.f0_hz.shape[0]) * (HOP_SAMPLES / SAMPLE_RATE),
            f0_hz=np.where(periodicity >= PERIODICITY_THRESHOLD, result.f0_hz, 0.0),
            confidence=np.clip(periodicity, 0.0, 1.0),
        )
