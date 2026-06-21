import re

# Whisper large-v3 was trained on large dumps of YouTube/OpenSubtitles captions,
# so on near-silent or instrumental regions it emits memorized caption boilerplate
# ("subtitle credits") that never occurs in song lyrics. These artifacts are
# language-specific but highly stereotyped; we drop any segment whose text is
# dominated by such boilerplate.
#
# References: known faster-whisper / openai-whisper hallucination corpora.

_HALLUCINATION_PATTERNS = [
    # Russian caption credits, e.g. "Субтитры создавал DimaTorzok",
    # "Продолжение следует", or editor credits like
    # "Редактор субтитров А.Семкин Корректор А.Егорова".  # noqa: RUF003
    r"субтитр",
    r"дима\s*торзок",
    r"dimatorzok",
    r"редактор\s+субтитров",
    r"корректор",
    r"продолжение\s+следует",
    r"подписывайтесь",
    r"спасибо\s+за\s+просмотр",
    # English caption credits.
    r"subtitles?\s+by",
    r"thanks?\s+for\s+watching",
    r"please\s+subscribe",
    r"subscribe\s+to",
    # Bare "Thank you" (one or more repeats, any casing/punctuation) emitted as a
    # standalone segment on a silent/fading outro — a classic Whisper caption
    # artifact. Anchored to the WHOLE segment so genuine lyrics that merely
    # contain "thank you" mid-line are left intact; the repeat group also catches
    # "Thank you. Thank you." chains.
    r"^(?:thank\s*you[\s.,!]*)+$",
    # Cross-language caption-platform signatures.
    r"amara\.org",
    r"transcri(?:bed|ption)\s+by",
    r"www\.",
    r"\.com\b",
    # Caption boilerplate emitted on instrumental / silent outros
    # (e.g. "© transcript", "Emily Beynon"): the copyright-transcript signature
    # never occurs in sung lyrics, and "Emily Beynon" is a well-known Whisper
    # credit hallucination.
    r"©\s*transcript",
    r"emily\s+beynon",
]

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in _HALLUCINATION_PATTERNS]


def _is_hallucination(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in _COMPILED)


def filter_hallucinated_segments(segments: list[dict]) -> list[dict]:
    return [
        segment
        for segment in segments
        if not _is_hallucination(segment.get("text", ""))
    ]
