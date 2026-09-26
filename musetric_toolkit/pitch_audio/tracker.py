from dataclasses import dataclass
from pathlib import Path

import librosa.sequence
import numpy as np
import torch
from torch.nn import functional

from musetric_toolkit.pitch_audio.rmvpe.model import E2E, N_BINS, N_MELS, MelSpectrogram

SAMPLE_RATE = 16000
WINDOW = 1024
MEL_FMIN = 30
MEL_FMAX = 8000
FRAME_MULTIPLE = 32
SEGMENT_FRAMES = 32000
SEGMENT_CONTEXT = 512
UNVOICED_SALIENCE = 0.03
TRANSITION_WIDTH = 12
CENTS_STEP = 20.0
CENTS_OFFSET = 1997.3794084376191
LOCAL_BINS = 4
BASE_HZ = 10.0


@dataclass(frozen=True)
class TrackerResult:
    f0_hz: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class Tracker:
    model: E2E
    mel: MelSpectrogram
    device: torch.device
    transition: np.ndarray
    cents_mapping: np.ndarray


def load_tracker(checkpoint_path: Path, hop_samples: int) -> Tracker:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = E2E(4, 1, (2, 2))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    mel = MelSpectrogram(
        N_MELS, SAMPLE_RATE, WINDOW, hop_samples, None, MEL_FMIN, MEL_FMAX
    )
    bins = np.arange(N_BINS)
    transition = np.maximum(TRANSITION_WIDTH - np.abs(bins[:, None] - bins[None, :]), 0)
    transition = transition.astype(np.float64)
    transition /= transition.sum(axis=1, keepdims=True)
    cents_mapping = np.pad(CENTS_STEP * bins + CENTS_OFFSET, (LOCAL_BINS, LOCAL_BINS))
    return Tracker(
        model=model.to(device),
        mel=mel.to(device),
        device=device,
        transition=transition,
        cents_mapping=cents_mapping,
    )


@torch.no_grad()
def _salience_segment(tracker: Tracker, mel: torch.Tensor) -> np.ndarray:
    n_frames = mel.shape[-1]
    padded = FRAME_MULTIPLE * ((n_frames - 1) // FRAME_MULTIPLE + 1)
    if padded > n_frames:
        mel = functional.pad(mel, (0, padded - n_frames), mode="constant")
    hidden = tracker.model(mel.float())[:, :n_frames]
    return hidden.squeeze(0).cpu().numpy()


@torch.no_grad()
def _salience(tracker: Tracker, audio_16k: np.ndarray) -> np.ndarray:
    audio = torch.from_numpy(audio_16k.astype(np.float32)).to(tracker.device)
    mel = tracker.mel(audio.unsqueeze(0))
    n_frames = mel.shape[-1]
    if n_frames <= SEGMENT_FRAMES + 2 * SEGMENT_CONTEXT:
        return _salience_segment(tracker, mel)
    parts = []
    for start in range(0, n_frames, SEGMENT_FRAMES):
        end = min(n_frames, start + SEGMENT_FRAMES)
        left = max(0, start - SEGMENT_CONTEXT)
        right = min(n_frames, end + SEGMENT_CONTEXT)
        segment = _salience_segment(tracker, mel[..., left:right])
        parts.append(segment[start - left : end - left])
    return np.concatenate(parts, axis=0)


def track(tracker: Tracker, audio_16k: np.ndarray) -> TrackerResult:
    salience = _salience(tracker, audio_16k)
    probabilities = salience.astype(np.float64).T
    probabilities /= probabilities.sum(axis=0, keepdims=True) + 1e-8
    path = librosa.sequence.viterbi(probabilities, tracker.transition).astype(np.int64)

    padded = np.pad(salience, ((0, 0), (LOCAL_BINS, LOCAL_BINS)))
    local = np.arange(2 * LOCAL_BINS + 1)[None, :] + path[:, None]
    rows = np.arange(padded.shape[0])[:, None]
    local_salience = padded[rows, local]
    local_cents = tracker.cents_mapping[local]
    cents = (local_salience * local_cents).sum(axis=1)
    cents /= local_salience.sum(axis=1) + 1e-12
    confidence = padded.max(axis=1)
    cents[confidence <= UNVOICED_SALIENCE] = 0.0
    f0_hz = BASE_HZ * np.power(2.0, cents / 1200.0)
    f0_hz[f0_hz == BASE_HZ] = 0.0
    return TrackerResult(
        f0_hz=f0_hz.astype(np.float64), confidence=confidence.astype(np.float64)
    )
