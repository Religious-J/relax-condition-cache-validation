"""Verify the public evidence packet using only the Python standard library."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evidence_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes evidence root: {relative}")
    return path


def verify(root: Path = ROOT, originals: Path | None = None) -> dict:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    seen = set()
    for row in manifest["files"]:
        relative = row["path"]
        if relative in seen:
            raise ValueError(f"Duplicate manifest path: {relative}")
        seen.add(relative)
        data = evidence_path(root, relative).read_bytes()
        if len(data) != row["public_bytes"] or digest(data) != row["public_sha256"]:
            raise ValueError(f"Public file hash/size differs: {relative}")
        content = gzip.decompress(data) if row["encoding"] == "gzip" else data
        if len(content) != row["public_content_bytes"] or digest(content) != row["public_content_sha256"]:
            raise ValueError(f"Public content hash/size differs: {relative}")
        if originals is not None:
            original = evidence_path(originals, row["source_path"]).read_bytes()
            if len(original) != row["source_bytes"] or digest(original) != row["source_sha256"]:
                raise ValueError(f"Original file hash/size differs: {relative}")
    actual = {str(p.relative_to(root)) for p in (root / "evidence").rglob("*") if p.is_file()}
    if seen != actual:
        raise ValueError(f"Unlisted or missing evidence: {sorted(seen ^ actual)}")
    migration = json.loads((root / "MIGRATION.json").read_text())
    copied = [row for row in migration["files"] if row.get("path")]
    for row in copied:
        data = evidence_path(root, row["path"]).read_bytes()
        if len(data) != row["bytes"] or digest(data) != row["sha256"]:
            raise ValueError(f"Migrated tool/test hash differs: {row['path']}")
    return {
        "public_files_verified": len(seen),
        "migrated_files_verified": len(copied),
        "original_files_verified": len(seen) if originals is not None else 0,
        "raw_tensor_comparisons_recomputed": False,
        "note": "Hashes check packet integrity, not independent authenticity of the recorded experiment.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-artifacts", type=Path, help="Optional private artifact root; never uploaded")
    args = parser.parse_args()
    print(json.dumps(verify(originals=args.original_artifacts), indent=2))


if __name__ == "__main__":
    main()
