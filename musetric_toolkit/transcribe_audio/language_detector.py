import collections
import logging

from whisperx.vads import Pyannote, Vad

# Uses WhisperX VAD internals for full-track language sampling.
# WhisperX is BSD-2-Clause; see thirdPartyNotices.md.

LANGUAGE_DETECTION_WINDOW_SECONDS = 20
MIN_SPEECH_SECONDS = 0.5


def detect_language_fulltrack(
    model,
    audio,
    sample_rate: int,
) -> str | None:
    total_samples = len(audio)
    if total_samples <= 0:
        return None
    windows = _build_vad_windows(model, audio, sample_rate)
    if not windows:
        windows = _build_fixed_windows(total_samples, sample_rate)
    languages = _collect_languages(model, audio, sample_rate, windows)
    return _pick_language(languages)


def _collect_languages(
    model,
    audio,
    sample_rate: int,
    windows: list[tuple[float, float]],
) -> list[str]:
    languages = []
    for start_seconds, end_seconds in windows:
        duration = max(0.0, end_seconds - start_seconds)
        if duration < MIN_SPEECH_SECONDS:
            continue
        clip = _slice_audio(audio, sample_rate, start_seconds, end_seconds)
        if clip is None:
            continue
        language = _detect_language_for_window(
            model,
            clip,
            start_seconds,
            end_seconds,
        )
        if language:
            languages.append(language)
    return languages


def _slice_audio(audio, sample_rate: int, start_seconds: float, end_seconds: float):
    start_sample = int(start_seconds * sample_rate)
    end_sample = int(end_seconds * sample_rate)
    clip = audio[start_sample:end_sample]
    if len(clip) == 0:
        return None
    return clip


def _detect_language_for_window(
    model,
    clip,
    start_seconds: float,
    end_seconds: float,
) -> str | None:
    try:
        return model.detect_language(clip)
    except Exception as error:
        logging.debug(
            "Language detection failed for %.2fs-%.2fs: %s",
            start_seconds,
            end_seconds,
            error,
        )
        return None


def _pick_language(languages: list[str]) -> str | None:
    if not languages:
        return None
    counts = collections.Counter(languages)
    most_common = counts.most_common()
    top_count = most_common[0][1]
    top_languages = [language for language, count in most_common if count == top_count]
    if len(top_languages) == 1:
        return top_languages[0]
    mid_language = languages[len(languages) // 2]
    if mid_language in top_languages:
        return mid_language
    return top_languages[0]


def _build_vad_windows(model, audio, sample_rate: int) -> list[tuple[float, float]]:
    vad_model = getattr(model, "vad_model", None)
    if vad_model is None:
        return []
    vad_params = getattr(model, "_vad_params", {}) or {}
    vad_onset = vad_params.get("vad_onset", 0.5)
    vad_offset = vad_params.get("vad_offset", 0.363)
    try:
        if issubclass(type(vad_model), Vad):
            waveform = vad_model.preprocess_audio(audio)
            merge_chunks = vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks
        vad_segments = vad_model({"waveform": waveform, "sample_rate": sample_rate})
        vad_segments = merge_chunks(
            vad_segments,
            LANGUAGE_DETECTION_WINDOW_SECONDS,
            onset=vad_onset,
            offset=vad_offset,
        )
    except Exception:
        return []
    windows = []
    for segment in vad_segments or []:
        start = segment.get("start")
        end = segment.get("end")
        if start is None or end is None:
            continue
        windows.append((float(start), float(end)))
    return windows


def _build_fixed_windows(
    total_samples: int,
    sample_rate: int,
) -> list[tuple[float, float]]:
    window_samples = int(LANGUAGE_DETECTION_WINDOW_SECONDS * sample_rate)
    if window_samples <= 0:
        return []
    if total_samples <= window_samples:
        return [(0.0, total_samples / sample_rate)]
    windows = []
    for start in range(0, total_samples, window_samples):
        end = min(start + window_samples, total_samples)
        windows.append((start / sample_rate, end / sample_rate))
    return windows
