# Whisper large-v3-turbo → word-timestamped ONNX (q4)

Exports **openai/whisper-large-v3-turbo** to the [transformers.js][tjs] ONNX layout
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
  --model_id openai/whisper-large-v3-turbo \
  --quantize --modes q4 \
  --output_attentions \
  --skip_validation \
  --output_parent_dir tmp/whisper-export
```

Output lands in `tmp/whisper-export/openai/whisper-large-v3-turbo/` with fp32 and q4
graphs under `onnx/` plus the tokenizer/config JSON. The runtime uses only:

```
config.json  generation_config.json  preprocessor_config.json
tokenizer.json  tokenizer_config.json  vocab.json  merges.txt
added_tokens.json  special_tokens_map.json  normalizer.json
onnx/encoder_model_q4.onnx  onnx/decoder_model_merged_q4.onnx
```

`stage_whisper.py` copies exactly those into `deps/whisper-large-v3-turbo-onnx/`
(the local publish repo).

## Make the encoder run on mobile GPUs (required)

Two independent defects stand between the stock export and a phone, and neither
is in the weights. `mobile_encoder.py` applies all three rewrites in order on a
single load of the graph, then checks that the result is actually clear:

```bash
uv run python scripts/onnx/whisper/mobile_encoder.py   --input  tmp/whisper-export/openai/whisper-large-v3-turbo/onnx/encoder_model_q4.onnx   --output tmp/whisper-export/openai/whisper-large-v3-turbo/onnx/encoder_model_q4.onnx
```

It is idempotent: a second run over an already-prepared graph reports that each
pass has nothing to do and writes the same bytes. `stage_whisper.py` runs the
same check before it copies anything, so an encoder that skipped this step
cannot reach the publish repo.

### The graph is too big to bind

The exported encoder computes each layer's attention in one shot, so its score
tensor is `(20, 1500, 1500)` fp32 - 171.7 MiB. WebGPU guarantees only 128 MiB
per storage binding and several mobile adapters offer exactly that minimum, so
the encoder cannot run there at all; ONNX Runtime fails the `Softmax` dispatch
with `binding index 1 not present in the bind group layout`.

`block_attention.py` splits the queries into blocks and concatenates the
per-block outputs. Softmax normalizes each query row over the full key axis on
its own, so this is exact, not an approximation. The default 250-row block caps
the score tensor at 28.6 MiB and grows the graph from 1751 to 2423 nodes. No
weight is touched - the q4 `MatMulNBits` nodes are copied through - and CPU
output is bit-identical to the unblocked graph. The pass refuses to write a
graph whose blocks would still exceed the 128 MiB floor.

### The graph computes the wrong answer

The second defect is not a limit but a silent wrong answer. On Adreno 600-series
adapters the WebGPU `Transpose` returns a quarter of its output as zeros. The
cause is in the execution provider's shader: it stages the tile in
`array<array<T, tile_size + 1>, tile_size>` - 16 x 17 f32, 1088 bytes - and on
that hardware a workgroup array whose size is not a multiple of 512 bytes loses
the stores of one warp out of four across `workgroupBarrier`. The stores land, a
thread reads its own slot back correctly, but the other threads do not see them.

Only the shapes that reach that shader are affected: after leading extents of
one are squeezed away, a rank-2 permutation swapping both axes. In this graph
that is one node out of 161 - plus both `Conv` nodes, because the provider
transposes NCHW to NHWC inside them. The full investigation, including the
one-line fix upstream, is in `plan/ort-webgpu-transpose-adreno660.md`.

`conv_to_matmul.py` writes the convolution as `Pad -> Slice x3 -> Concat ->
Reshape -> MatMul -> Add -> Reshape` with the weight on the left, so no
transpose appears at all. `transpose_to_gather.py` replaces the remaining
affected node with a `Gather` against a precomputed index table - 7.3 MiB of
int32 for this graph - and leaves the other 160 transposes alone.

Together: 2423 -> 2441 nodes and CPU output unchanged to 9e-5, while the
affected adapter goes from visibly wrong output to what a healthy one reports.

The index table is static, so the rewrite pins the encoder to batch 1 - which is
what the runtime feeds anyway, one 30-second window at a time.

### Running a single pass

Each pass is still its own script with the same `--input`/`--output` interface,
which is what you want when bisecting a graph or trying a different block size:

```bash
uv run python scripts/onnx/whisper/block_attention.py --input X --output Y --query-block 250
uv run python scripts/onnx/whisper/conv_to_matmul.py --input X --output Y
uv run python scripts/onnx/whisper/transpose_to_gather.py --input X --output Y
```

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
cd deps/whisper-large-v3-turbo-onnx
git init -b main && git lfs install --local && git lfs track "*.onnx"
git add -A
git -c user.name="Your Name" -c user.email="you@example.com" \
  commit -m "Whisper large-v3-turbo word-timestamped ONNX (q4, flat)"
git remote add origin https://huggingface.co/musetric/whisper-large-v3-turbo-onnx
git push -f origin main   # auth via the token from `hf auth login`
```

[tjs]: https://github.com/huggingface/transformers.js
[tjsconv]: https://github.com/huggingface/transformers.js/tree/main/scripts
