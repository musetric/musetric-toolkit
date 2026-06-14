import json
from pathlib import Path

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.key_audio.response_builder import build_payload
from musetric_toolkit.key_audio.skey_runner import run_skey


def main(args) -> None:
    send_message({"type": "progress", "progress": 0.0})

    root, mode, confidence = run_skey(args.audio_path, args.models_path)
    send_message({"type": "progress", "progress": 0.9})

    payload = build_payload(root, mode, confidence)

    output_path = Path(args.result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as result_file:
        json.dump(payload, result_file, ensure_ascii=False)

    send_message({"type": "progress", "progress": 1.0})
