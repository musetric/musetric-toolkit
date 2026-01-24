import json
import logging
import re
import sys
from pathlib import Path

from musetric_toolkit.common.logger import redirect_std_streams, setup_logging
from musetric_toolkit.transcribe_audio.cli import parse_arguments
from musetric_toolkit.transcribe_audio.lyric_splitter import (
    split_segments_by_lyrics,
)
from musetric_toolkit.transcribe_audio.response_builder import (
    build_payload_segments,
)
from musetric_toolkit.transcribe_audio.whisperx_runner import (
    transcribe_with_whisperx,
)


def build_stream_suppression_patterns(
    log_level: str,
) -> list[re.Pattern[str]]:
    if log_level == "debug":
        return []
    return [
        re.compile(r"\bwhisperx\.[\w\.]+ - INFO - "),
        re.compile(r"^Model was trained with "),
    ]


def main() -> None:
    args = parse_arguments()
    setup_logging(args.log_level)
    redirect_std_streams(
        suppress_patterns=build_stream_suppression_patterns(args.log_level)
    )

    try:
        segments, _language = transcribe_with_whisperx(args.audio_path, args.log_level)
        lyric_segments = split_segments_by_lyrics(segments)
        payload_segments = build_payload_segments(lyric_segments)
        output_path = Path(args.result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as result_file:
            json.dump(payload_segments, result_file, ensure_ascii=False)
    except Exception as error:
        logging.error("Audio transcription failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
