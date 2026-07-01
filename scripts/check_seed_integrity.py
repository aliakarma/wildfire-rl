#!/usr/bin/env python
"""Seed-integrity guard: fail if "independent" seeds are actually degenerate.

Two degeneracies the audit found and this guard blocks:
  1. Distinct seed labels that map to byte-identical checkpoints.
  2. Distinct-seed PPO rows in an eval CSV with byte-identical metric values (e.g. seed_2 == seed_3),
     which understates variance and inflates apparent n.

Exit code 0 = OK, 1 = degeneracy detected. CI-callable (Phase 13) and part of `make reproduce`
(Phase 6). Missing artifacts are skipped, so it is safe to run before the Phase 9 retrain.

    python scripts/check_seed_integrity.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_models(pattern: str = "ppo_*seed_*.zip") -> list[str]:
    paths = sorted(Path("models").glob(pattern))
    if not paths:
        print(f"  [models] no checkpoints matching '{pattern}' (skip)")
        return []
    by_hash: dict[str, list[str]] = {}
    for p in paths:
        by_hash.setdefault(_sha256(p), []).append(p.name)
    dupes = [names for names in by_hash.values() if len(names) > 1]
    if dupes:
        return [f"identical checkpoints across seeds: {dupes}"]
    print(f"  [models] {len(paths)} checkpoints, all distinct")
    return []


def check_eval_csv(csv: str, key: str = "reward_mean") -> list[str]:
    df = pd.read_csv(csv)
    if "policy" not in df.columns or key not in df.columns:
        return []
    ppo = df[df["policy"].astype(str).str.startswith("ppo")]
    if len(ppo) < 2:
        return []
    n_dup = int(ppo[key].round(6).duplicated().sum())
    if n_dup:
        return [f"{csv}: {n_dup} identical PPO per-seed '{key}' rows (degenerate seeds)"]
    print(f"  [eval] {csv}: {len(ppo)} PPO rows, all distinct")
    return []


def main() -> int:
    print("Seed-integrity check:")
    problems: list[str] = []
    problems += check_models()
    for csv in (
        "results/eval_saudi.csv",
        "results/eval_california.csv",
        "results/eval_saudi_generalization.csv",
        "results/eval_california_multiseed_california.csv",
    ):
        if Path(csv).exists():
            problems += check_eval_csv(csv)

    if problems:
        print("SEED INTEGRITY: FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("SEED INTEGRITY: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
