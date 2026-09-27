from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.parity_audio.case_writer import CaseWriter
from musetric_toolkit.parity_audio.host_audio import POWER, read_mono

# The transcription step at the level of one chunk: the host sends the lead stem
# as the power downmix at 16 kHz, and the product decodes a chunk of up to 30 s
# with the pinned ONNX pair through transformers.js
# (packages/ai/src/runtime/whisper/whisperDecoder.ts): log-mel features padded
# to 30 s, the q4 encoder, then greedy decoding with timestamps by the fp16
# merged decoder and its cache, with a token budget, and a time for every token
# from the alignment heads' cross-attention cropped to the chunk. The original
# is openai/whisper-large-v3-turbo in torch, and the author runs its `generate`
# with the model's own defaults. The decoder is compared under teacher forcing,
# one token per step with the cache as the product runs it, so that a token
# that differs on a device does not change what the later steps see.

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 30 * SAMPLE_RATE
ENCODER_POSITION_SAMPLES = 320
ENCODER_POSITIONS = 1500
MAX_DECODE_TOKENS = 400
MIN_DECODE_TOKENS = 32
TOKENS_PER_SECOND = 12
END_OF_TEXT = 50257
ORIGINAL_ID = "openai/whisper-large-v3-turbo"
ORIGINAL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
DECODER_FILE = "decoder_model_merged_fp16.onnx"


def js_round(value: float) -> int:
    return int(np.floor(value + 0.5))


def decode_budget(samples: int) -> tuple[int, int]:
    """The product's encoder positions of the chunk and its token budget."""
    positions = min(ENCODER_POSITIONS, js_round(samples / ENCODER_POSITION_SAMPLES))
    seconds = samples / SAMPLE_RATE
    budget = min(
        MAX_DECODE_TOKENS,
        max(
            MIN_DECODE_TOKENS, js_round(seconds * TOKENS_PER_SECOND) + MIN_DECODE_TOKENS
        ),
    )
    return positions, budget


def cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def generated_tokens(tokens: list[int], timestamp_begin: int) -> list[int]:
    """Tokens from the first timestamp on, without the closing end of text."""
    start = next(
        (index for index, token in enumerate(tokens) if token >= timestamp_begin),
        len(tokens),
    )
    kept = tokens[start:]
    return kept[:-1] if kept and kept[-1] == END_OF_TEXT else kept


def timed_tokens(output, timestamp_begin: int) -> np.ndarray:
    """[tokens, 2]: each generated token and its time, rounded as the product does."""
    tokens = [int(token) for token in output["sequences"][0].tolist()]
    times = [float(time) for time in output["token_timestamps"][0].tolist()]
    start = next(
        (index for index, token in enumerate(tokens) if token >= timestamp_begin),
        len(tokens),
    )
    kept = generated_tokens(tokens, timestamp_begin)
    rows = [
        (token, js_round(times[start + index] * 100) / 100)
        for index, token in enumerate(kept)
    ]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def teacher_forced_onnx(
    session: ort.InferenceSession,
    encoder_states: np.ndarray,
    forced: list[int],
    prompt_length: int,
) -> np.ndarray:
    """Logits of every position but the last: the prompt in one step without a
    cache, then one token per step, as transformers.js runs the merged decoder."""
    past = {}
    for model_input in session.get_inputs():
        if model_input.name.startswith("past_key_values."):
            _, heads, _, width = model_input.shape
            past[model_input.name] = np.zeros((1, heads, 0, width), dtype=np.float16)
    names = [output.name for output in session.get_outputs()]
    states = encoder_states.astype(np.float32)

    def step(ids: list[int], use_cache: bool) -> dict[str, np.ndarray]:
        values = session.run(
            names,
            {
                **past,
                "input_ids": np.asarray([ids], dtype=np.int64),
                "encoder_hidden_states": states,
                "use_cache_branch": np.asarray([use_cache]),
            },
        )
        return dict(zip(names, values, strict=True))

    outputs = step(forced[:prompt_length], use_cache=False)
    logits = [outputs["logits"][0]]
    for name in past:
        past[name] = outputs[name.replace("past_key_values.", "present.")]
    for position in range(prompt_length, len(forced) - 1):
        outputs = step([forced[position]], use_cache=True)
        logits.append(outputs["logits"][0])
        for name in past:
            if ".decoder." in name:
                past[name] = outputs[name.replace("past_key_values.", "present.")]
    return np.concatenate(logits).astype(np.float64)


def load_original(device: torch.device):
    from transformers import WhisperForConditionalGeneration  # noqa: PLC0415

    model = WhisperForConditionalGeneration.from_pretrained(
        ORIGINAL_ID, revision=ORIGINAL_REVISION, torch_dtype=torch.float32
    )
    return model.to(device).eval()


def run(args, writer: CaseWriter) -> None:
    from transformers import (  # noqa: PLC0415
        GenerationConfig,
        WhisperFeatureExtractor,
    )

    bundle = Path(args.onnx).parent
    language = args.language
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    audio = read_mono(args.audio_path, SAMPLE_RATE, POWER)[:CHUNK_SAMPLES]
    samples = audio.shape[0]
    positions, budget = decode_budget(samples)
    extractor = WhisperFeatureExtractor.from_pretrained(bundle)
    features = extractor(
        audio.astype(np.float32), sampling_rate=SAMPLE_RATE, return_tensors="np"
    ).input_features
    config = GenerationConfig.from_pretrained(bundle)
    timestamp_begin = config.no_timestamps_token_id + 1
    prompt = [
        config.decoder_start_token_id,
        config.lang_to_id[f"<|{language}|>"],
        config.task_to_id["transcribe"],
    ]
    writer.tensor("unit.input", "reference", audio)
    writer.tensor("model.input", "reference", features)

    send_message({"type": "progress", "progress": 0.1})
    model = load_original(device)
    features_t = torch.from_numpy(features).to(device)
    with torch.inference_mode():
        encoded = model.model.encoder(features_t).last_hidden_state
    writer.tensor("encoder.output", "original", encoded.double().cpu().numpy())
    encoder = cpu_session(Path(args.onnx))
    (onnx_encoded,) = encoder.run(["last_hidden_state"], {"input_features": features})
    writer.tensor("encoder.output", "onnx-cpu", onnx_encoded)

    send_message({"type": "progress", "progress": 0.4})
    reference = model.generate(
        input_features=features_t,
        generation_config=config,
        language=language,
        task="transcribe",
        return_timestamps=True,
        return_token_timestamps=True,
        max_new_tokens=budget,
        num_frames=2 * positions,
    )
    tokens = [int(token) for token in reference["sequences"][0].tolist()]
    forced = prompt + generated_tokens(tokens, timestamp_begin) + [END_OF_TEXT]
    writer.tensor(
        "result.tokens", "reference", timed_tokens(reference, timestamp_begin)
    )
    writer.tensor("decoder.tokens", "reference", np.asarray(forced), "int32")
    writer.meta.update(
        {
            "sampleRate": SAMPLE_RATE,
            "samples": samples,
            "language": language,
            "positions": positions,
            "maxNewTokens": budget,
            "promptLength": len(prompt),
            "timestampBegin": timestamp_begin,
        }
    )

    send_message({"type": "progress", "progress": 0.6})
    with torch.inference_mode():
        logits = model(
            input_features=features_t,
            decoder_input_ids=torch.tensor([forced[:-1]], device=device),
        ).logits
    writer.tensor("decoder.logits", "original", logits[0].double().cpu().numpy())
    decoder = cpu_session(bundle / DECODER_FILE)
    writer.tensor(
        "decoder.logits",
        "onnx-cpu",
        teacher_forced_onnx(decoder, onnx_encoded, forced, len(prompt)),
    )

    send_message({"type": "progress", "progress": 0.8})
    author = model.generate(
        input_features=features_t,
        language=language,
        task="transcribe",
        return_timestamps=True,
        return_token_timestamps=True,
    )
    writer.tensor("result.tokens", "author", timed_tokens(author, timestamp_begin))
