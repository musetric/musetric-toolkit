import os
import sys


def default_models_path() -> str:
    if os.name == "nt":
        base = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or os.path.expanduser("~")
        )
        return os.path.join(base, "musetric", "models")
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
        return os.path.join(base, "musetric", "models")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "musetric", "models")
