"""Phase 8: emit LaTeX table rows for the revised manuscript from certified artifacts.

Prints (to stdout, for pasting into the .tex):
  [TABLE 1]  main results, both regions, all reported policies, WEL/ISR with 95% CIs
  [APP B]    real per-seed WEL for the hierarchy (fixes the fabricated C6 table)
  [APP C]    Welch (t, df, p, d) + TOST equivalence for the key comparisons
  [ABLATION] Saudi ablation rows with CIs + vs-full-system p
No value is hand-transcribed; everything is read from results/phase{2,3,4}_peer/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

P2 = Path("results/phase2_peer")
P3 = Path("results/phase3_peer")
P4 = Path("results/phase4_peer")


def esc(s: str) -> str:
    return s.replace("&", "\\&")


# Display order + label for Table 1
ORDER = [
    ("No-Op", "No-Op"),
    ("Greedy-Risk Heuristic", "Greedy-Risk"),
    ("Local Reactive", "Local Reactive"),
    ("Flat MARL (matched)", "Flat MARL (MAPPO)"),
    ("QMIX (matched)", "QMIX"),
    ("CommNet (matched)", "CommNet"),
    ("Value-First Heuristic", "Value-First (expert)"),
    ("Learned Hierarchical", "Hierarchical (ours)"),
]


def load_region(region: str) -> dict:
    s = json.load(open(P2 / f"eval_summary_{region}.json"))["metrics"]
    react = pd.read_csv(P2 / f"multiseed_eval_aggregate_{region}_reactive.csv")
    rv = react.loc[react.policy == "Local Reactive", "WEL"].to_numpy()
    ri = react.loc[react.policy == "Local Reactive", "ISR"].to_numpy()
    import numpy as np

    s["WEL"]["Local Reactive"] = {
        "mean": float(rv.mean()), "ci_lo": float(np.percentile(
            np.random.default_rng(0).choice(rv, (10000, 5)).mean(1), 2.5)),
        "ci_hi": float(np.percentile(
            np.random.default_rng(0).choice(rv, (10000, 5)).mean(1), 97.5)),
    }
    s["ISR"]["Local Reactive"] = {"mean": float(ri.mean())}
    return s


print("% ===================== [TABLE 1] main results =====================")
for region, disp in [("saudi", "Saudi Arabia"), ("california", "California")]:
    s = load_region(region)
    best_wel = min(s["WEL"][k]["mean"] for k, _ in ORDER)
    print(f"% --- {disp} ---")
    for key, label in ORDER:
        w = s["WEL"][key]
        i = s["ISR"][key]
        wel = f"{w['mean']:.2f}"
        ci = ""
        if "ci_lo" in w and abs(w["ci_hi"] - w["ci_lo"]) > 1e-6:
            ci = f" \\small{{[{w['ci_lo']:.1f}, {w['ci_hi']:.1f}]}}"
        isr = f"{i['mean']:.3f}"
        bold_w = f"\\textbf{{{wel}}}" if abs(w["mean"] - best_wel) < 1e-6 else wel
        print(f"& {label} & {bold_w}{ci} & {isr} \\\\")

print("\n% ===================== [APP B] real per-seed WEL =====================")
for region, disp in [("saudi", "Saudi Arabia"), ("california", "California")]:
    s = json.load(open(P2 / f"eval_summary_{region}.json"))["metrics"]["WEL"]
    ps = s["Learned Hierarchical"]["per_seed"]
    vf = s["Value-First Heuristic"]["per_seed"]
    print(f"Hierarchical, {disp} & " + " & ".join(f"{x:.1f}" for x in ps) + " \\\\")
    print(f"Value-First, {disp} & " + " & ".join(f"{x:.1f}" for x in vf) + " \\\\")

print("\n% ===================== [APP C] Welch + TOST =====================")
st = json.load(open(P4 / "stats_tables.json"))
tost = json.load(open(P4 / "tost_results.json"))


def welch_row(label, d):
    if d["p"] is None:
        return f"{label} & --- & --- & degenerate & --- \\\\"
    dd = f"{d['cohens_d']:.2f}" if d["cohens_d"] is not None else "---"
    return f"{label} & {d['t']:.2f} & {d['df']:.1f} & {d['p']:.4g} & {dd} \\\\"


for region, disp in [("saudi", "Saudi"), ("california", "California")]:
    wel = st[region]["WEL"]
    print(f"% --- {disp}: Hierarchical vs ... (WEL) ---")
    print(welch_row(f"{disp}: Hier vs Value-First",
                    wel["Value-First Heuristic"]["vs_Hierarchical"]))
    print(welch_row(f"{disp}: Hier vs Flat MARL",
                    wel["Flat MARL (matched)"]["vs_Hierarchical"]))
    print(welch_row(f"{disp}: CommNet vs Hier",
                    wel["CommNet (matched)"]["vs_Hierarchical"]))

print("\n% TOST equivalence (pre-registered +/-5% margin):")
for k, v in tost.items():
    if isinstance(v, dict):
        print(f"%   {k}: diff {v['mean_diff']:+.3f}, margin +/-{v['margin']:.3f}, "
              f"p_TOST {v['p_tost']:.4f}, equivalent={v['equivalent_at_0.05']}")

print("\n% ===================== [ABLATION] Saudi =====================")
ab = json.load(open(P3 / "ablation_summary_saudi.json"))
wel = ab["metrics"]["WEL"]
cmp = ab["comparisons_vs_full_shipped"]["WEL"]
order = [
    "Full System (shipped ckpt)", "BC-only (no RL fine-tuning)", "w/o BC Pretraining",
    "w/o KL Regularization", "w/o Entropy Bonus", "w/o WEL/ISR Reward", "w/o Target-Seeking",
]
for k in order:
    m = wel[k]
    ci = f"\\small{{[{m['ci_lo']:.1f}, {m['ci_hi']:.1f}]}}"
    if k == "Full System (shipped ckpt)":
        p = "---"
    else:
        c = cmp.get(k, {})
        p = "degenerate" if c.get("p") is None else f"{c['p']:.3g}"
    print(f"{esc(k)} & {m['mean']:.2f} {ci} & {m['mean'] and ab['metrics']['ISR'][k]['mean']:.3f} & {p} \\\\")
