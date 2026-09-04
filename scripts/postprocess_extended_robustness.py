"""Normalize the extended robustness sweep's medium column to the main-protocol evaluation.

The frozen 4-policy robustness artifact (``results/wildfire_phase4/``) was built by resuming from a
CSV seeded with the main-results evaluation, so its ``medium`` cells (medium == the
``default`` regime) carry the 15-episode Table-1 values while ``easy``/``hard`` were
computed at 10 episodes per seed group. The extended 7-policy sweep computes every new
cell at 10 episodes; for the medium column that would silently mix evaluation budgets
across policies within one column.

This script rewrites ``results/wildfire_phase4_extended/robustness_summary.json`` so that every
policy's ``medium`` cell equals its frozen main-results (15-episode) WEL/ISR mean from
``results/wildfire_phase3_multiseed/phase3_summary.json`` — the identical environment condition,
since regime ``medium`` is regime ``default``. Easy/hard cells are left as computed
(10 episodes per seed group, uniformly for all 7 policies). The summary records the
episode budget per regime so downstream consumers need not guess.

Run after ``run_phase4_ablations.py --study robustness --out results/wildfire_phase4_extended``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "results" / "wildfire_phase4_extended"
PHASE3 = REPO / "results" / "wildfire_phase3_multiseed"

POLICIES = [
    "noop",
    "value_first",
    "greedy_risk",
    "local_reactive",
    "mappo",
    "commnet",
    "hiercomm_heur",
]


def main() -> None:
    summary = json.loads((EXT / "robustness_summary.json").read_text())
    phase3 = json.loads((PHASE3 / "phase3_summary.json").read_text())["regions"]

    replaced = 0
    for region, block in summary["regions"].items():
        for policy in POLICIES:
            if policy not in block.get("medium", {}):
                continue
            m = phase3[region][policy]["metrics"]
            cell = block["medium"][policy]
            target = {"WEL_mean": m["WEL"]["mean"], "ISR_mean": m["ISR"]["mean"]}
            if cell != target:
                block["medium"][policy] = target
                replaced += 1

    summary["protocol"]["episodes_by_regime"] = {
        "easy": 10,
        "medium": "main protocol (15; medium == default regime, values from phase3_summary.json)",
        "hard": 10,
    }
    (EXT / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"medium cells normalized to the main 15-episode evaluation: {replaced} replaced")

    for region, block in summary["regions"].items():
        for regime in ("easy", "medium", "hard"):
            cells = block[regime]
            best = min(cells, key=lambda p: cells[p]["WEL_mean"])
            print(f"  {region}/{regime}: best={best} ({cells[best]['WEL_mean']:.2f})")


if __name__ == "__main__":
    main()
