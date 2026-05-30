# Musetric Toolkit

Standalone CLI tool extracted from [Musetric](https://github.com/popelenkow/musetric) so it can be installed from GitHub releases and run worker scripts directly from the terminal.

## Installation

Install the package directly from the latest GitHub release. Pick the `cuda` or `cpu` extra to match the PyTorch index — it selects the matching `onnxruntime-gpu` / `onnxruntime` distribution.
```bash
uv tool install --python 3.13.2 \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cu129 \  # /whl/cpu
  --index-strategy unsafe-best-match \
  --overrides https://raw.githubusercontent.com/popelenkow/musetric-toolkit/main/overrides.txt \
  "musetric-toolkit[cuda] @ https://github.com/popelenkow/musetric-toolkit/releases/download/v0.0.19/musetric_toolkit-0.0.19-py3-none-any.whl"  # [cpu]
```

For local development, install the CLI in editable mode.
```bash
uv tool install --python 3.13.2 --editable ".[cuda]" \  ".[cpu]"
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cu129 \  # /whl/cpu
  --index-strategy unsafe-best-match \
  --overrides overrides.txt
```

The `cpu` / `cuda` extras select the `onnxruntime` (CPU) vs `onnxruntime-gpu` (CUDA) package — they share the `onnxruntime` import name but are distinct PyPI packages, so the `--index` flag alone cannot switch them the way it switches `torch`.

The `overrides.txt` file pins `torch` and `torchaudio` to 2.8.0, overriding `skey`'s overly conservative `<2.8.0` upper bound. `uv tool install` does not read `[tool.uv]` from the target project, so the CLI flag is required.

## CLI Usage

```bash
musetric-separate \
  --source-path /path/to/input.wav \  # input audio file
  --lead-path /path/to/output-lead.flac \  # output path for the lead vocal track
  --backing-path /path/to/output-backing.flac \  # output path for the backing vocal track
  --instrumental-path /path/to/output-instrumental.flac \  # output path for the instrumental track
  --sample-rate 44100 \  # target sample rate (e.g. 44100)
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

```bash
musetric-transcribe \
  --audio-path /path/to/vocals.wav \  # input vocal audio file
  --result-path /path/to/transcription.json \  # output JSON file
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

## Dependencies

### BSRoformer Neural Network

- **Source:** https://github.com/lucidrains/BS-RoFormer by Phil Wang (MIT)
- **Usage:** Audio source separation model (adapted)
- **Thanks to:** https://github.com/nomadkaraoke/python-audio-separator (MIT) — research tool that helped validate the BSRoformer approach and integration patterns

### WhisperX Speech Transcription

- **Source:** https://github.com/m-bain/whisperX by Max Bain (MIT)
- **Usage:** Speech-to-text + word-level alignment for `musetric-transcribe`

## License

Musetric Toolkit is [MIT licensed](https://github.com/popelenkow/Musetric/blob/main/license.md).
