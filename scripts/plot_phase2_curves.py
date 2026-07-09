"""Plot Phase 2 (peer-review remediation) training curves.

Produces results/phase2_peer/train_curves.png: rolling-mean episode return and episode WEL
vs. environment steps for MAPPO / QMIX / CommNet on both regions, with the No-Op WEL level
as a reference line (convergence evidence required by the Phase 2 gate, issue H1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("results/phase2_peer")
ALGOS = ["mappo", "qmix", "commnet"]
REGIONS = ["saudi", "california"]
NOOP_WEL = {"saudi": 27.0, "california": 25.25}  # certified Table 1 No-Op level
WINDOW = 25

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
for r, region in enumerate(REGIONS):
    for algo in ALGOS:
        df = pd.read_csv(BASE / f"train_curve_{algo}_{region}.csv")
        roll_ret = df["episode_return"].rolling(WINDOW, min_periods=5).mean()
        roll_wel = df["WEL"].rolling(WINDOW, min_periods=5).mean()
        axes[r, 0].plot(df["env_steps"], roll_ret, label=algo.upper())
        axes[r, 1].plot(df["env_steps"], roll_wel, label=algo.upper())
    axes[r, 1].axhline(
        NOOP_WEL[region], color="gray", ls="--", lw=1, label="No-Op WEL (certified)"
    )
    axes[r, 0].set_ylabel(f"{region.capitalize()}\nepisode return (rolling {WINDOW})")
    axes[r, 1].set_ylabel(f"episode WEL (rolling {WINDOW})")
    axes[r, 0].grid(alpha=0.3)
    axes[r, 1].grid(alpha=0.3)

axes[1, 0].set_xlabel("environment steps")
axes[1, 1].set_xlabel("environment steps")
axes[0, 0].set_title("Training return (WEL/ISR-matched objective)")
axes[0, 1].set_title("Training-episode WEL (lower is better)")
axes[0, 0].legend()
axes[0, 1].legend()
fig.suptitle(
    "Phase 2 rehabilitated baselines - 100k env steps, matched objective, seed 42",
    fontsize=11,
)
fig.tight_layout()
out = BASE / "train_curves.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")

# Plateau check: compare mean WEL of first vs last training quintile
for region in REGIONS:
    for algo in ALGOS:
        df = pd.read_csv(BASE / f"train_curve_{algo}_{region}.csv")
        q = len(df) // 5
        first, last = df["WEL"][:q].mean(), df["WEL"][-q:].mean()
        print(f"{region:11s} {algo:8s} WEL first-quintile {first:6.2f} -> last {last:6.2f}"
              f"  (return {df['episode_return'][:q].mean():7.2f} -> "
              f"{df['episode_return'][-q:].mean():7.2f})")
