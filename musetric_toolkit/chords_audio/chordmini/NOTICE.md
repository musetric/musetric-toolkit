# Vendored: ChordMini

This directory vendors the **inference subset** of ChordMini.

- Upstream: https://github.com/ptnghia-j/ChordMini
- License: MIT (see `LICENSE` in this directory).
- Vendored on: 2026-05-29.

## What was copied

- `models/` — verbatim from upstream `src/models/` (ChordNet + BTC + shared transformer blocks).
- `utils/` — verbatim from upstream `src/utils/`, minus `dataloader.py` and
  `gradient_utils.py` (training-only, not on the inference path).
- `ChordMini.yaml` — verbatim from upstream `config/ChordMini.yaml`.

## What was changed

- Absolute imports rooted at `src.` were rewritten to
  `musetric_toolkit.chords_audio.chordmini.` so the subset imports as a
  normal package. No logic was modified.

## What was NOT copied

Upstream `src/evaluation/`, `src/training/`, `src/training_scripts/`,
`src/data/`, notebooks, and the ChordMiniApp. The inference feature/predict
helpers we need (CQT extraction, sliding-window prediction) are reimplemented
in `../chordmini_runner.py`, which avoids upstream's `seaborn` / `matplotlib`
/ `mir_eval` (evaluation-only) dependencies.

The `2e1d_model_best.pth` checkpoint is downloaded at runtime (see
`../chordmini_checkpoint.py`); it is not vendored.
