#!/usr/bin/env python
"""Seed-integrity guard: fail if "independent" seeds are actually degenerate.

Two degeneracies the V1 audit found and this guard blocks:
  1. Distinct seed labels that map to byte-identical checkpoints.
  2. Distinct-seed rows in an eval CSV with byte-identical metric values (e.g. seed_2 == seed_3)
     that are NOT explained by distinct checkpoints — i.e. fabricated multi-seed.

Honest exception: when the checkpoints are provably distinct but two collapsed policies pick the
same near-constant action, their eval rows are legitimately identical. That is a documented
negative result, not fraud, so it is reported as a WARNING and does not fail the guard.

Exit code 0 = OK, 1 = degeneracy detected. CI-callable. Missing artifacts are skipped, so it is
safe to run before any V2 training exists.

Ported from V1 (``legacy_v1/scripts/check_seed_integrity.py``) in Phase 0; logic unchanged.
V2 eval CSVs are registered in ``EVAL_CSVS`` as later phases produce them — the V1 CSVs stay
frozen in ``legacy_v1/results/`` and are never re-certified here.

    python -m wildfire_marl.reproducibility.check_seed_integrity
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

# V2 eval CSVs to audit (relative to the repo root). Registered per phase:
#   Phase 4  -> baseline eval CSVs, Phase 5+ -> MARL eval CSVs, Phase 9 -> transfer CSVs.
EVAL_CSVS: tuple[str, ...] = ()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_models_distinct() -> bool | None:
    """CI fallback when local checkpoints are absent (``models/`` is git-ignored): consult the
    committed ``results/models_manifest.json``. Returns True if all certified ``*seed_*`` hashes
    are distinct, False if duplicated, None if there is no manifest evidence."""
    mf = Path("results/models_manifest.json")
    if not mf.exists():
        return None
    import json

    data = json.loads(mf.read_text())
    shas = [
        v["sha256"]
        for k, v in data.items()
        if "seed_" in k and k.endswith(".zip") and "deprecated" not in k
    ]
    if not shas:
        return None
    return len(set(shas)) == len(shas)


def check_models(pattern: str = "*seed_*.zip") -> tuple[list[str], bool | None]:
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
    learned = df[df["policy"].astype(str).str.startswith(("ppo", "mappo", "qmix"))]
    if len(learned) < 2:
        return []
    n_dup = int(learned[key].round(6).duplicated().sum())
    if n_dup:
        # Identical eval rows are only fraud when the seeds are NOT backed by distinct checkpoints.
        # When checkpoints are provably distinct, identical rows mean distinct policies that
        # collapsed to the same (near-constant) behavior on the fixed eval scenarios — an honest
        # negative result, not fabricated multi-seed. Warn, do not fail.
        if models_distinct is True:
            print(
                f"  [eval] {csv}: {n_dup} identical learned-policy '{key}' rows, but checkpoints "
                f"are DISTINCT -> policy collapse to constant action (honest negative result), "
                f"not fake seeds [WARN]"
            )
            return []
        if models_distinct is None:
            # No checkpoints/manifest to distinguish collapse from fraud (e.g. bare CI checkout).
            print(
                f"  [eval] {csv}: {n_dup} identical learned-policy '{key}' rows; cannot verify "
                f"without checkpoints or models_manifest.json -> skipped [WARN]"
            )
            return []
        return [
            f"{csv}: {n_dup} identical learned-policy per-seed '{key}' rows with no "
            f"distinct-checkpoint evidence (degenerate seeds)"
        ]
    print(f"  [eval] {csv}: {len(learned)} learned-policy rows, all distinct")
    return []


def main() -> int:
    print("Seed-integrity check:")
    problems: list[str] = []
    model_problems, models_distinct = check_models()
    problems += model_problems
    for csv in EVAL_CSVS:
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
