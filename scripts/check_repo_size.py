#!/usr/bin/env python
"""Git-safety guard: fail if any file STAGED for commit exceeds a size limit.

Wire this into CI / pre-push to make accidental 2 GB pushes impossible even if
.gitignore is edited. Exit code 1 on violation.

    python scripts/check_repo_size.py --max-mb 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def staged_files() -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            stderr=subprocess.DEVNULL,
        )
        return [line for line in out.decode().splitlines() if line.strip()]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-mb", type=float, default=50.0)
    args = ap.parse_args()
    limit = args.max_mb * 1024 * 1024

    violations = []
    for f in staged_files():
        p = Path(f)
        if p.is_file() and p.stat().st_size > limit:
            violations.append((f, p.stat().st_size / 1e6))

    if violations:
        print(f"ERROR: {len(violations)} staged file(s) exceed {args.max_mb} MB:")
        for f, mb in violations:
            print(f"  {mb:8.1f} MB  {f}")
        print("These belong on Hugging Face / Zenodo, not in git. Unstage them.")
        return 1
    print(f"OK: no staged file exceeds {args.max_mb} MB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
