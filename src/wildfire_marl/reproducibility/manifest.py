"""Data/model integrity manifests (sha256) for reproducibility snapshots.

Since large artifacts (tensors, checkpoints, raw data) live outside git, a checksum
manifest lets anyone verify they pulled the exact bytes a result was produced from.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    root: str | Path, patterns: Iterable[str] = ("*.npy", "*.zip", "*.tif", "*.nc", "*.csv")
) -> dict[str, dict[str, object]]:
    """Walk ``root`` and record {relative_path: {sha256, bytes}} for matching files."""
    root = Path(root)
    manifest: dict[str, dict[str, object]] = {}
    for pattern in patterns:
        for f in sorted(root.rglob(pattern)):
            if f.is_file():
                rel = f.relative_to(root).as_posix()
                manifest[rel] = {"sha256": sha256_file(f), "bytes": f.stat().st_size}
    return manifest


def write_manifest(root: str | Path, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(build_manifest(root), indent=2))
    return out_path


def verify_manifest(root: str | Path, manifest_path: str | Path) -> list[str]:
    """Return a list of mismatches (missing or changed files). Empty list == OK."""
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text())
    problems: list[str] = []
    for rel, info in manifest.items():
        f = root / rel
        if not f.exists():
            problems.append(f"MISSING: {rel}")
        elif sha256_file(f) != info["sha256"]:
            problems.append(f"CHANGED: {rel}")
    return problems
