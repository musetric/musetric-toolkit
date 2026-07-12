"""Stage the published Whisper file set from an export dir into the deps repo.

`convert.py` writes many graphs (fp32 + q4, all decoder variants); the publish
repo needs only the q4 encoder + merged decoder plus the tokenizer/config JSON.
This copies exactly that set into `deps/whisper-large-v3-onnx/` (the folder
`publish_whisper.py` uploads).

    uv run python scripts/onnx/whisper/stage_whisper.py \
      --export tmp/whisper-export/openai/whisper-large-v3
"""

# ruff: noqa: T201

import argparse
import contextlib
import shutil
import sys
from pathlib import Path

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

# Repo layout: <root>/music/musetric-toolkit/scripts/onnx/whisper/stage_whisper.py
# and <root>/deps/whisper-large-v3-onnx — so parents[5] is <root>.
DEFAULT_DEST = Path(__file__).resolve().parents[5] / "deps" / "whisper-large-v3-onnx"
DEFAULT_EXPORT = Path("tmp/whisper-export/openai/whisper-large-v3")

PUBLISH_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "encoder_model_q4.onnx",
    "decoder_model_merged_q4.onnx",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    export: Path = args.export
    dest: Path = args.dest
    missing = [rel for rel in PUBLISH_FILES if not (export / rel).is_file()]
    if missing:
        listing = "\n  ".join(missing)
        raise SystemExit(f"missing exported files under {export}:\n  {listing}")

    for rel in PUBLISH_FILES:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".onnx"):
            shutil.copy2(export / rel, target)
        else:
            # Normalize text to LF so the local files match what Hugging Face
            # serves (the hub normalizes text blobs to LF on commit); keeps the
            # sha256 manifest and the @musetric/ai descriptor consistent.
            target.write_bytes((export / rel).read_bytes().replace(b"\r\n", b"\n"))
        print(f"staged {rel}")
    print(f"staged {len(PUBLISH_FILES)} files into {dest}")


if __name__ == "__main__":
    main()
