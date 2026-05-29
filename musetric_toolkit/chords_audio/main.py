import json
from pathlib import Path

from musetric_toolkit.chords_audio.chordmini_checkpoint import ensure_checkpoint
from musetric_toolkit.chords_audio.chordmini_runner import run_chordmini
from musetric_toolkit.chords_audio.response_builder import build_payload
from musetric_toolkit.common.logger import send_message


def main(args) -> None:
    send_message({"type": "progress", "progress": 0.0})

    checkpoint_path = ensure_checkpoint(args.models_path)
    predictions, frame_duration, idx_to_chord = run_chordmini(
        args.audio_path,
        checkpoint_path,
    )

    payload = build_payload(predictions, frame_duration, idx_to_chord)

    output_path = Path(args.result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as result_file:
        json.dump(payload, result_file, ensure_ascii=False)

    send_message({"type": "progress", "progress": 1.0})
