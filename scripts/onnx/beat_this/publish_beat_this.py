"""Publish the Beat This! rhythm ONNX repo to Hugging Face.

Uploads the staged folder (see stage_beat_this.py): the `beat_this.onnx` tracker
(spectrogram windows -> beat/downbeat logits), the `mel-filterbank.bin` its
WebGPU front end projects onto, their `config.json` descriptor, and the README
model card. The `@musetric/ai` runtime runs the graph on onnxruntime-web's
WebGPU execution provider.

Usage (requires `hf auth login` with write access to the musetric org):

    uv run python scripts/onnx/beat_this/publish_beat_this.py --src <publish-dir>

Pass --dry-run to verify the file set and print the sha256 manifest without
uploading.
"""

# ruff: noqa: T201

import argparse
import contextlib
import hashlib
import sys
from pathlib import Path

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

PUBLISH_FILES = [
    "beat_this.onnx",
    "config.json",
    "mel-filterbank.bin",
]

DEFAULT_REPO = "musetric/beat-this-onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(src: Path) -> list[tuple[str, int, str]]:
    missing = [rel for rel in PUBLISH_FILES if not (src / rel).is_file()]
    if missing:
        listing = "\n  ".join(missing)
        raise SystemExit(f"missing expected files under {src}:\n  {listing}")
    manifest: list[tuple[str, int, str]] = []
    for rel in PUBLISH_FILES:
        path = src / rel
        manifest.append((rel, path.stat().st_size, sha256(path)))
    return manifest


def print_manifest(manifest: list[tuple[str, int, str]]) -> None:
    total = 0
    print(
        "sha256                                                            "
        "     size  file"
    )
    for rel, size, digest in manifest:
        total += size
        print(f"{digest}  {size:>11}  {rel}")
    print(f"total: {total / 1e6:.1f} MB across {len(manifest)} files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify files + print the sha256 manifest, upload nothing",
    )
    args = parser.parse_args()

    src: Path = args.src
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")

    manifest = collect(src)
    print(f"source: {src}")
    print_manifest(manifest)

    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    print(f"uploading {len(manifest)} files (+ README.md) to {args.repo} ...")
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(src),
        allow_patterns=[*PUBLISH_FILES, "README.md"],
        commit_message="Add Beat This! rhythm ONNX (log-mel + beat tracker)",
    )
    print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
