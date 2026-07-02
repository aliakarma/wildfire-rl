#!/usr/bin/env python
"""Effective-method gate: the method the paper claims is effective must actually beat no-op.

Reframed (Phase 16 finding): rigorous evaluation showed that fully-corrected single-agent PPO does
NOT beat no-op on this task, while the heuristic routers (nearest_fire / frontier) contain the fire
near-perfectly. So the gate now verifies:

  1. The BEST reported policy beats no-op by ``margin`` on burned cells (a working method exists).
  2. It reports PPO's honest status separately (PPO is a NEGATIVE-result baseline, not the method).

This preserves the original purpose — block a false "our method works" claim — under the honest
framing where the heuristic router is the effective method. Exit 0 = a working method is present.

    python scripts/validate_learning_gate.py            # default 5% margin
    python scripts/validate_learning_gate.py --margin 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EVAL_CSVS = ["results/eval_saudi.csv", "results/eval_california.csv"]


def gate(csv: str, margin: float) -> bool | None:
    df = pd.read_csv(csv)
    if "policy" not in df.columns or "burned_cells_mean" not in df.columns:
        print(f"  {csv}: no policy/burned_cells columns (skip)")
        return None
    noop = df.loc[df["policy"].astype(str) == "noop", "burned_cells_mean"]
    non_noop = df.loc[df["policy"].astype(str) != "noop"]
    if noop.empty or non_noop.empty:
        print(f"  {csv}: missing noop or comparison rows (skip)")
        return None

    noop_m = float(noop.mean())
    threshold = noop_m * (1.0 - margin)
    best_row = non_noop.loc[non_noop["burned_cells_mean"].idxmin()]
    best_policy, best_m = str(best_row["policy"]), float(best_row["burned_cells_mean"])
    ok = best_m <= threshold
    print(
        f"  {csv}: effective method = '{best_policy}' burned={best_m:.2f}  "
        f"noop={noop_m:.2f}  need <= {threshold:.2f}  -> {'PASS' if ok else 'FAIL'}"
    )

    # Honest reporting of PPO as a negative-result baseline.
    ppo = df.loc[df["policy"].astype(str).str.startswith("ppo"), "burned_cells_mean"]
    if not ppo.empty:
        ppo_m = float(ppo.mean())
        verdict = "beats no-op" if ppo_m <= threshold else "does NOT beat no-op (negative result)"
        print(f"      PPO burned={ppo_m:.2f} -> {verdict}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--margin", type=float, default=0.05, help="required fractional improvement vs noop"
    )
    args = ap.parse_args()

    print(
        f"Effective-method gate (some reported method must cut burned cells >= {args.margin:.0%} vs noop):"
    )
    results = [gate(c, args.margin) for c in EVAL_CSVS if Path(c).exists()]
    results = [r for r in results if r is not None]

    if not results:
        print("EFFECTIVE-METHOD GATE: NO DATA -> FAIL")
        return 1
    if all(results):
        print("EFFECTIVE-METHOD GATE: PASS (a working method beats no-op)")
        return 0
    print("EFFECTIVE-METHOD GATE: FAIL — no reported method beats no-op by the required margin.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
