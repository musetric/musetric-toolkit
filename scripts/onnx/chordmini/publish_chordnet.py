"""Publish the ChordNet/CQT artifact bundle to Hugging Face.

Validates and uploads the staged artifact produced by ``stage_chordnet.py``:

* ``chordnet.onnx``;
* its ``config.json`` classifier contract;
* ``cqt-plan.bin``; and
* ``cqt-plan.manifest.json``.

The CQT plan travels with the ONNX file so a published model has an explicit,
hashable feature-extraction contract.  The generated TypeScript payload and
manifest source files deliberately do not belong in this bundle.

    uv run python scripts/onnx/chordmini/publish_chordnet.py --src <publish-dir>

Pass ``--dry-run`` to verify the file set and print its sha256 manifest without
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
    "config.json",
    "chordnet.onnx",
    "cqt-plan.bin",
    "cqt-plan.manifest.json",
]

DEFAULT_REPO = "musetric/chordmini-onnx"


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it whole."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(src: Path) -> list[tuple[str, int, str]]:
    """Verify a staged artifact and build its deterministic size/hash manifest."""
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
    """Print hashes in a compact, copyable form for release review."""
    total = 0
    print(
        "sha256                                                            "
        "     size  file"
    )
    for rel, size, digest in manifest:
        total += size
        print(f"{digest}  {size:>11}  {rel}")
    print(f"total: {total / 1e6:.0f} MB across {len(manifest)} files")


def main() -> None:
    """Validate a staged artifact and, unless dry-run, upload it to the Hub."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument(
        "--repo",
        help=f"Hugging Face model repo; defaults to {DEFAULT_REPO}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify files + print the sha256 manifest, upload nothing",
    )
    args = parser.parse_args()

    src: Path = args.src
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")

    repo = args.repo or DEFAULT_REPO
    manifest = collect(src)
    print(f"source: {src}")
    print(f"repo: {repo}")
    print_manifest(manifest)

    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    print(f"uploading {len(manifest)} files (+ README.md) to {repo} ...")
    api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(src),
        allow_patterns=[*PUBLISH_FILES, "README.md"],
        commit_message="Add ChordNet ONNX and CQT plan artifacts",
    )
    print(f"done: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
