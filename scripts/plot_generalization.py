"""Phase 5: replacement for the retired TRS heatmap (Figure 4).

Grouped-bar chart of dWEL (improvement over same-condition No-Op) per policy per held-out
condition, both regions. No-Op sits at 0 by construction; the horizontal line at 0 makes
the falsifiability visible — several strategic policies fall ON the line under rotated
layouts / cross-region, which the old TRS (all >= 0.95) could never show.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("results/phase5_peer")
CONDITIONS = ["heldout_ignition", "wind_plus90", "assets_rot90", "cross_region"]
COND_LABEL = {
    "heldout_ignition": "Held-out\nignitions",
    "wind_plus90": "Wind +90°",
    "assets_rot90": "Rotated\nassets",
    "cross_region": "Cross-region\ntransfer",
}
POLICIES = ["Value-First Heuristic", "Local Reactive", "Learned Hierarchical", "CommNet (matched)"]
COLORS = {
    "Value-First Heuristic": "#888888",
    "Local Reactive": "#4C9F70",
    "Learned Hierarchical": "#3B6EA5",
    "CommNet (matched)": "#C0504D",
}


def dwel_for(summary: dict, cond: str, policy: str) -> float:
    pols = summary["conditions"].get(cond, {})
    for name, e in pols.items():
        base = name.split(" [trained on")[0]
        if base == policy:
            return e["dWEL_vs_noop"]
    return np.nan


fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, region in zip(axes, ["saudi", "california"], strict=True):
    s = json.load(open(BASE / f"generalization_summary_{region}.json"))
    x = np.arange(len(CONDITIONS))
    w = 0.2
    for i, pol in enumerate(POLICIES):
        vals = [dwel_for(s, c, pol) for c in CONDITIONS]
        ax.bar(x + (i - 1.5) * w, vals, w, label=pol, color=COLORS[pol])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS], fontsize=9)
    ax.set_title(f"{region.capitalize()}")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("ΔWEL vs No-Op  (higher = adds value; 0 = no better than inaction)")
axes[1].legend(fontsize=8, loc="upper right")
fig.suptitle(
    "Held-out generalization (replaces the vacuous TRS metric): strategic policies fall to "
    "0 under rotated layouts and cross-region transfer",
    fontsize=10,
)
fig.tight_layout()
out = BASE / "generalization_dwel.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
