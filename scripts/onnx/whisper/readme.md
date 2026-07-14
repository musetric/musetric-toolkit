# Whisper large-v3 → word-timestamped ONNX (q4)

Exports **openai/whisper-large-v3** to the [transformers.js][tjs] ONNX layout
with the cross-attention **alignment heads** baked into `generation_config.json`,
so the decoder emits word-level timestamps. Quantized to **q4** and published to
Hugging Face for the `@musetric/ai` (`packages/ai`) WebGPU runtime.

This is a different toolchain from `../roformer/` (which exports the separation
core from the toolkit's own torch model). Here `convert.py`, `quantize.py` and
`extra/whisper.py` are **vendored from [huggingface/transformers.js][tjsconv]**
(Apache-2.0); only the whisper code path is exercised.

## Environment

Pinned deps conflict with the toolkit's own `transformers`, so the export runs
in an **isolated** uv environment (never `uv sync` these into the project):

- `requirements.txt` — the pinned export stack (Python 3.11).

## Export

Run from the toolkit root. `--output_attentions` selects the custom ONNX config
that outputs cross-attentions; the task is left as `auto` (whisper resolves to
`automatic-speech-recognition-with-past`, which the alignment-head config needs).

```bash
PYTHONPATH=scripts/onnx uv run --no-project --python 3.11 \
  --with-requirements scripts/onnx/whisper/requirements.txt \
  python -m whisper.convert \
  --model_id openai/whisper-large-v3 \
  --quantize --modes q4 \
  --output_attentions \
  --skip_validation \
  --output_parent_dir tmp/whisper-export
```

Output lands in `tmp/whisper-export/openai/whisper-large-v3/` with fp32 and q4
graphs under `onnx/` plus the tokenizer/config JSON. The runtime uses only:

```
config.json  generation_config.json  preprocessor_config.json
tokenizer.json  tokenizer_config.json  vocab.json  merges.txt
added_tokens.json  special_tokens_map.json  normalizer.json
onnx/encoder_model_q4.onnx  onnx/decoder_model_merged_q4.onnx
```

`stage_whisper.py` copies exactly those into `deps/whisper-large-v3-onnx/`
(the local publish repo).

## Publish

`publish_whisper.py` verifies the staged files, prints a sha256 manifest,
creates the HF repo and uploads the folder including its `README.md` model card.
Requires `hf auth login` with write access to the musetric org.

```bash
# preview: verify + sha256 manifest, upload nothing
uv run python scripts/onnx/whisper/publish_whisper.py --dry-run
# publish
uv run python scripts/onnx/whisper/publish_whisper.py
```

The printed sha256 manifest feeds the `@musetric/ai` whisper model descriptor
(`packages/ai/src/models/whisperModel.ts`), which downloads and checksum-verifies
these files the same way the separation core is fetched. The repo is a **flat**
layout (the q4 graphs sit next to the config, not under `onnx/`); the runtime
loads them with `subfolder: ''`.

`publish_whisper.py` uses the HF API, which records the commit author as your
Hugging Face no-reply address. To make the author match the committer (your git
email), publish over git instead:

```bash
cd deps/whisper-large-v3-onnx
git init -b main && git lfs install --local && git lfs track "*.onnx"
git add -A
git -c user.name="Your Name" -c user.email="you@example.com" \
  commit -m "Whisper large-v3 word-timestamped ONNX (q4, flat)"
git remote add origin https://huggingface.co/musetric/whisper-large-v3-onnx
git push -f origin main   # auth via the token from `hf auth login`
```

[tjs]: https://github.com/huggingface/transformers.js
[tjsconv]: https://github.com/huggingface/transformers.js/tree/main/scripts
