def build_payload_words(words: list) -> list:
    payload_words = []
    for word in words or []:
        text = (word.get("word") or "").strip()
        start = word.get("start")
        end = word.get("end")
        if not text or start is None or end is None:
            continue
        payload_words.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
            }
        )
    return payload_words


def build_fallback_word(text: str, start: float, end: float) -> dict:
    return {
        "start": start,
        "end": end,
        "text": text,
    }


def build_payload_segments(segments: list) -> list:
    payload_segments = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment["start"])
        end = float(segment["end"])
        payload_words = build_payload_words(segment.get("words", []))
        if not payload_words:
            payload_words = [build_fallback_word(text, start, end)]
        payload_segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "words": payload_words,
            }
        )
    return payload_segments
