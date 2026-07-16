"""Generate the librosa CQT reference table used by @musetric/cqt tests.

`@musetric/cqt` reproduces `librosa.cqt`, so its tests need librosa's numbers.
Rather than committing reference arrays as binary assets, this script reduces
the comparison to one scalar per bin: the CQT peak magnitude of a pure tone
sitting on that bin's centre frequency.

That scalar is what a binary fixture would otherwise be committed for. It is
stable (the interior frames of a steady tone agree to ~1e-5), it pins the
absolute scale of the transform - so a wrong `norm`, `scale`, gain or window
cannot pass - and it is small enough to live in TypeScript, where the tests can
read it.

Bin mapping, silence, superposition and linearity are asserted analytically in
`cqt.test.ts` and need nothing from here.

    uv run python scripts/onnx/chordmini/generate_cqt_reference.py \
      --out <packages/cqt/src/cqt/__test__/reference.ts>
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 22050
HOP_LENGTH = 2048
FMIN = 32.70319566257483
N_BINS = 144
BINS_PER_OCTAVE = 24

TONE_AMPLITUDE = 0.5
TONE_SECONDS = 4

# One bin at each end of every octave, plus both ends of the range, so a
# mis-mapped octave boundary or a reversed bin order cannot pass.
TONE_BINS = [0, 12, 23, 24, 47, 48, 71, 72, 95, 96, 119, 120, 143]


def _bin_frequency(bin_index: int) -> float:
    return FMIN * 2 ** (bin_index / BINS_PER_OCTAVE)


def _tone(frequency: float) -> np.ndarray:
    sample_count = SAMPLE_RATE * TONE_SECONDS
    time = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    return np.asarray(
        TONE_AMPLITUDE * np.sin(2 * np.pi * frequency * time), dtype="<f4"
    )


def _peak_magnitude(bin_index: int) -> float:
    cqt = np.abs(
        librosa.cqt(
            _tone(_bin_frequency(bin_index)),
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            fmin=FMIN,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
            tuning=0,
            filter_scale=1,
            norm=1,
            sparsity=0.01,
            window="hann",
            scale=True,
            pad_mode="constant",
            res_type="soxr_hq",
        )
    )
    column = cqt[:, cqt.shape[1] // 2]
    peak_bin = int(np.argmax(column))
    if peak_bin != bin_index:
        raise SystemExit(
            f"librosa put bin {bin_index} at {peak_bin}; refusing to write"
        )
    return float(column[bin_index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [(bin_index, _peak_magnitude(bin_index)) for bin_index in TONE_BINS]
    entries = "\n".join(
        f"  {{ bin: {bin_index}, peakMagnitude: {magnitude:.6f} }},"
        for bin_index, magnitude in rows
    )
    content = f"""export type CqtToneReference = {{
  bin: number;
  peakMagnitude: number;
}};

export const cqtToneAmplitude = {TONE_AMPLITUDE};
export const cqtToneSeconds = {TONE_SECONDS};

export const cqtToneReferences: CqtToneReference[] = [
{entries}
];
"""
    args.out.write_text(content, encoding="utf-8")
    for bin_index, magnitude in rows:
        frequency = _bin_frequency(bin_index)
        print(f"bin {bin_index:3d} ({frequency:8.2f} Hz) -> {magnitude:.6f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
