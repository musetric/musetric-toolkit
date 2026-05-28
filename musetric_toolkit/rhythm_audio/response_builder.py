import numpy as np


def build_payload(
    beats: np.ndarray,
    downbeats: np.ndarray,
    bpm: float,
    meter: int,
) -> dict:
    return {
        "bpm": float(bpm),
        "beats": [float(t) for t in beats.tolist()],
        "downbeats": [float(t) for t in downbeats.tolist()],
        "meter": int(meter),
    }
