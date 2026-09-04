"""Freeze a results directory for provenance.

Writes a SHA-256 ``MANIFEST.sha256`` over every file in the directory and a single
``FREEZE.json`` fingerprint (hash of the manifest) plus the git commit, so the exact frozen
numbers are tamper-evident and reproducible. Re-running with the same inputs yields the same
fingerprint.

    python scripts/freeze_results.py results/wildfire_phase3_multiseed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

_SKIP = {"MANIFEST.sha256", "FREEZE.json"}


def sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze a results directory (SHA-256 manifest).")
    ap.add_argument("results_dir")
    args = ap.parse_args()
    root = Path(args.results_dir)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in _SKIP)
    manifest = "".join(f"{sha256(p)}  {p.relative_to(root).as_posix()}\n" for p in files)
    (root / "MANIFEST.sha256").write_text(manifest)
    fingerprint = hashlib.sha256(manifest.encode()).hexdigest()

    meta = {
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_date": str(date.today()),
        "results_dir": root.name,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "n_files": len(files),
        "fingerprint_sha256": fingerprint,
    }
    (root / "FREEZE.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"\nFROZEN {len(files)} files | fingerprint {fingerprint[:16]}...  -> {root}/FREEZE.json")


if __name__ == "__main__":
    main()
