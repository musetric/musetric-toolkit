import numpy as np

_NO_CHORD_LABEL = "N"
_UNKNOWN_LABEL = "X"


def _split_label(label: str) -> tuple[str, str | None]:
    if label in (_NO_CHORD_LABEL, _UNKNOWN_LABEL):
        return label, None
    if ":" in label:
        root, quality = label.split(":", 1)
        return root, quality
    return label, "maj"


def build_payload(
    predictions: np.ndarray,
    frame_duration: float,
    idx_to_chord: dict,
) -> dict:
    segments: list[dict] = []
    if len(predictions) == 0:
        return {"segments": segments}

    def label_at(index: int) -> str:
        return str(idx_to_chord.get(int(index), idx_to_chord.get(169, _NO_CHORD_LABEL)))

    def append_segment(start: float, end: float, index: int) -> None:
        label = label_at(index)
        root, quality = _split_label(label)
        segments.append(
            {
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "label": label,
                "root": root,
                "quality": quality,
            }
        )

    prev = int(predictions[0])
    start_time = 0.0
    for i in range(1, len(predictions)):
        cur = int(predictions[i])
        if cur != prev:
            end_time = i * frame_duration
            append_segment(start_time, end_time, prev)
            start_time = end_time
            prev = cur

    append_segment(start_time, len(predictions) * frame_duration, prev)
    return {"segments": segments}
