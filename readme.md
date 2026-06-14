# Musetric Toolkit

Standalone CLI tool extracted from [Musetric](https://github.com/popelenkow/musetric) so its worker scripts can run directly from the terminal.

## Installation

The toolkit is CUDA-only: it pulls the CUDA 12.9 PyTorch build and
`onnxruntime-gpu`, and is supported on Linux x86_64 and Windows (NVIDIA GPU).
Install the CLI in editable mode:

```bash
uv tool install --python 3.13.11 --editable "." \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

`onnxruntime-gpu` exposes the CUDA / DirectML execution providers and falls back
to the CPU provider when no GPU is present.

## CLI Usage

Separate audio into lead vocals, backing vocals, and instrumental parts.

```bash
musetric-separate \
  --source-path /path/to/input.wav \  # input audio file
  --lead-path /path/to/output-lead.flac \  # output path for the lead vocal track
  --backing-path /path/to/output-backing.flac \  # output path for the backing vocal track
  --instrumental-path /path/to/output-instrumental.flac \  # output path for the instrumental track
  --sample-rate 48000 \  # target sample rate
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

Transcribe vocals with WhisperX.

```bash
musetric-transcribe \
  --audio-path /path/to/vocals.wav \  # input vocal audio file
  --result-path /path/to/transcription.json \  # output JSON file
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

Analyze rhythm (BPM, beats, downbeats) with beat_this.

```bash
musetric-rhythm \
  --audio-path /path/to/input.wav \  # input audio file
  --result-path /path/to/rhythm.json \  # output JSON file (BPM, beats, downbeats)
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

Detect the musical key (root + mode) with S-KEY.

```bash
musetric-key \
  --audio-path /path/to/input.wav \  # input audio file
  --result-path /path/to/key.json \  # output JSON file (root, mode, confidence)
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

Detect chords with timings using ChordMini.

```bash
musetric-chords \
  --audio-path /path/to/input.wav \  # input audio file
  --result-path /path/to/chords.json \  # output JSON file (chord segments with timings)
  --models-path /path/to/models \  # base directory for downloaded models
  --log-level info  # debug|info|warn|error (default: info)
```

## License

Musetric Toolkit is [MIT licensed](https://github.com/popelenkow/Musetric/blob/main/license.md).
Third-party notices are listed in [thirdPartyNotices.md](thirdPartyNotices.md).
