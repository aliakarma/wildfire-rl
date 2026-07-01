#!/usr/bin/env python
"""Learning gate: PPO must actually beat doing nothing, or the run is not publishable.

Reads the per-region eval CSVs and requires the mean PPO ``burned_cells`` to be at least
``MARGIN`` below the ``noop`` baseline. This is the machine-checkable answer to the audit's
central finding (PPO was byte-identical to no-op). Exit 0 = PASS, 1 = FAIL.

    python scripts/validate_learning_gate.py            # default 5% margin
    python scripts/validate_learning_gate.py --margin 0.5

Before the Phase 9 retrain + Phase 16 strengthening this is EXPECTED to fail (the committed
CSVs still hold the collapsed policy). It becomes a blocking gate once real training lands.
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
    ppo = df.loc[df["policy"].astype(str).str.startswith("ppo"), "burned_cells_mean"]
    if noop.empty or ppo.empty:
        print(f"  {csv}: missing noop or ppo rows (skip)")
        return None
    noop_m, ppo_m = float(noop.mean()), float(ppo.mean())
    threshold = noop_m * (1.0 - margin)
    ok = ppo_m <= threshold
    print(
        f"  {csv}: ppo={ppo_m:.2f}  noop={noop_m:.2f}  "
        f"need ppo <= {threshold:.2f}  -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--margin", type=float, default=0.05, help="required fractional improvement vs noop"
    )
    args = ap.parse_args()

    print(f"Learning gate (PPO must reduce burned cells by >= {args.margin:.0%} vs noop):")
    results = [gate(c, args.margin) for c in EVAL_CSVS if Path(c).exists()]
    results = [r for r in results if r is not None]

    if not results:
        print("LEARNING GATE: NO DATA (no eval CSVs with ppo/noop rows) -> FAIL")
        return 1
    if all(results):
        print("LEARNING GATE: PASS")
        return 0
    print("LEARNING GATE: FAIL — PPO does not beat no-op by the required margin.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
