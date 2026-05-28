import json
from pathlib import Path

import numpy as np

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.rhythm_audio.beat_this_runner import run_beat_this
from musetric_toolkit.rhythm_audio.bpm_estimator import (
    estimate_bpm,
    estimate_meter,
)
from musetric_toolkit.rhythm_audio.response_builder import build_payload


def main(args) -> None:
    send_message({"type": "progress", "progress": 0.0})

    beats, downbeats = run_beat_this(args.audio_path)
    send_message({"type": "progress", "progress": 0.9})

    beats_arr = np.asarray(beats)
    downbeats_arr = np.asarray(downbeats)
    bpm = estimate_bpm(beats_arr)
    meter = estimate_meter(beats_arr, downbeats_arr)

    payload = build_payload(beats_arr, downbeats_arr, bpm, meter)

    output_path = Path(args.result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as result_file:
        json.dump(payload, result_file, ensure_ascii=False)

    send_message({"type": "progress", "progress": 1.0})
