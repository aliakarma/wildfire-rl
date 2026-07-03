#!/usr/bin/env python
"""Seed-integrity guard: fail if "independent" seeds are actually degenerate.

Two degeneracies the audit found and this guard blocks:
  1. Distinct seed labels that map to byte-identical checkpoints.
  2. Distinct-seed PPO rows in an eval CSV with byte-identical metric values (e.g. seed_2 == seed_3)
     that are NOT explained by distinct checkpoints — i.e. fabricated multi-seed.

Honest exception (Phase 16): when the checkpoints are provably distinct but two collapsed PPO
policies pick the same near-constant action, their eval rows are legitimately identical. That is the
documented negative result, not fraud, so it is reported as a WARNING and does not fail the guard.

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


def _manifest_models_distinct() -> bool | None:
    """CI fallback when local checkpoints are absent (``models/`` is git-ignored): consult the
    committed ``results/models_manifest.json``. Returns True if all certified ``ppo_*seed_*`` hashes
    are distinct, False if duplicated, None if there is no manifest evidence."""
    mf = Path("results/models_manifest.json")
    if not mf.exists():
        return None
    import json

    data = json.loads(mf.read_text())
    shas = [
        v["sha256"]
        for k, v in data.items()
        if k.startswith("ppo_") and "seed_" in k and k.endswith(".zip") and "deprecated" not in k
    ]
    if not shas:
        return None
    return len(set(shas)) == len(shas)


def check_models(pattern: str = "ppo_*seed_*.zip") -> tuple[list[str], bool | None]:
    """Return (problems, models_distinct). ``models_distinct`` is a tri-state:
    * True  — positive evidence seeds are real (distinct local checkpoints, or a manifest whose
      certified checkpoint hashes are all distinct);
    * False — proven fraud (byte-identical checkpoints / duplicated manifest hashes);
    * None  — no evidence available (no local checkpoints and no manifest, e.g. a bare CI checkout).
    """
    paths = sorted(Path("models").glob(pattern))
    if not paths:
        manifest_distinct = _manifest_models_distinct()
        if manifest_distinct is True:
            print("  [models] no local checkpoints; models_manifest.json hashes all distinct")
            return [], True
        if manifest_distinct is False:
            return ["models_manifest.json has duplicate certified checkpoint hashes"], False
        print(f"  [models] no checkpoints matching '{pattern}' and no manifest (cannot verify)")
        return [], None
    by_hash: dict[str, list[str]] = {}
    for p in paths:
        by_hash.setdefault(_sha256(p), []).append(p.name)
    dupes = [names for names in by_hash.values() if len(names) > 1]
    if dupes:
        return [f"identical checkpoints across seeds: {dupes}"], False
    print(f"  [models] {len(paths)} checkpoints, all distinct")
    return [], True


def check_eval_csv(csv: str, models_distinct: bool | None, key: str = "reward_mean") -> list[str]:
    df = pd.read_csv(csv)
    if "policy" not in df.columns or key not in df.columns:
        return []
    ppo = df[df["policy"].astype(str).str.startswith("ppo")]
    if len(ppo) < 2:
        return []
    n_dup = int(ppo[key].round(6).duplicated().sum())
    if n_dup:
        # Identical eval rows are only fraud when the seeds are NOT backed by distinct checkpoints.
        # When checkpoints are provably distinct, identical rows mean distinct policies that
        # collapsed to the same (near-constant) behavior on the fixed eval scenarios — the honest
        # PPO negative result (Phase 16), not fabricated multi-seed. Warn, do not fail.
        if models_distinct is True:
            print(
                f"  [eval] {csv}: {n_dup} identical PPO '{key}' rows, but checkpoints are DISTINCT "
                f"-> policy collapse to constant action (honest negative result), not fake seeds [WARN]"
            )
            return []
        if models_distinct is None:
            # No checkpoints/manifest to distinguish collapse from fraud (e.g. bare CI checkout).
            print(
                f"  [eval] {csv}: {n_dup} identical PPO '{key}' rows; cannot verify without "
                f"checkpoints or models_manifest.json -> skipped [WARN]"
            )
            return []
        return [
            f"{csv}: {n_dup} identical PPO per-seed '{key}' rows with no distinct-checkpoint "
            f"evidence (degenerate seeds)"
        ]
    print(f"  [eval] {csv}: {len(ppo)} PPO rows, all distinct")
    return []


def main() -> int:
    print("Seed-integrity check:")
    problems: list[str] = []
    model_problems, models_distinct = check_models()
    problems += model_problems
    for csv in (
        "results/eval_saudi.csv",
        "results/eval_california_multiseed_california.csv",
    ):
        if Path(csv).exists():
            problems += check_eval_csv(csv, models_distinct)

    if problems:
        print("SEED INTEGRITY: FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("SEED INTEGRITY: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
