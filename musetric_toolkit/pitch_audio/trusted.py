import numpy as np

CONFIDENCE_CORE = 0.5
CONFIDENCE_EXTEND = 0.1
MAX_STEP_CENTS = 150.0
ENERGY_WINDOW_SECONDS = 0.064
ENERGY_CORE_PERCENTILE = 10.0
ENERGY_RELATIVE_FLOOR_DB = 12.0
ENERGY_ABSOLUTE_FLOOR_DB = -55.0
ENERGY_MAX_FLOOR_DB = -42.0


def _frame_rms_db(
    audio: np.ndarray, sample_rate: int, hop_samples: int, n_frames: int
) -> np.ndarray:
    window = max(3, round(sample_rate * ENERGY_WINDOW_SECONDS))
    if window % 2 == 0:
        window += 1
    power = np.asarray(audio, dtype=np.float64) ** 2
    mean_power = np.convolve(power, np.ones(window) / window, mode="same")
    positions = np.minimum(
        np.arange(n_frames) * hop_samples, max(0, audio.shape[0] - 1)
    )
    return 20.0 * np.log10(np.sqrt(mean_power[positions] + 1e-12) + 1e-12)


def energy_gate(
    audio: np.ndarray,
    sample_rate: int,
    hop_samples: int,
    f0_hz: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    voiced = f0_hz > 0.0
    if not voiced.any():
        return voiced
    rms_db = _frame_rms_db(audio, sample_rate, hop_samples, f0_hz.shape[0])
    core = voiced & (confidence > CONFIDENCE_CORE)
    anchor = rms_db[core] if core.any() else rms_db[voiced]
    floor = float(
        np.percentile(anchor, ENERGY_CORE_PERCENTILE) - ENERGY_RELATIVE_FLOOR_DB
    )
    floor = float(np.clip(floor, ENERGY_ABSOLUTE_FLOOR_DB, ENERGY_MAX_FLOOR_DB))
    return voiced & (rms_db >= floor)


def _joins(
    f0_hz: np.ndarray, confidence: np.ndarray, index: int, neighbor: int
) -> bool:
    if f0_hz[index] <= 0.0 or confidence[index] < CONFIDENCE_EXTEND:
        return False
    step = abs(1200.0 * np.log2(f0_hz[index] / f0_hz[neighbor]))
    return bool(step <= MAX_STEP_CENTS)


def trusted_mask(f0_hz: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    trusted = (f0_hz > 0.0) & (confidence > CONFIDENCE_CORE)
    count = trusted.shape[0]
    for index in range(1, count):
        if not trusted[index] and trusted[index - 1]:
            trusted[index] = _joins(f0_hz, confidence, index, index - 1)
    for index in range(count - 2, -1, -1):
        if not trusted[index] and trusted[index + 1]:
            trusted[index] = _joins(f0_hz, confidence, index, index + 1)
    return trusted
