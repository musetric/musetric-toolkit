def build_payload(root: str, mode: str, confidence: float) -> dict:
    return {
        "root": root,
        "mode": mode,
        "confidence": float(confidence),
    }
