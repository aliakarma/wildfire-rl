"""Phase-6 cross-environment validation + failure modes (Task 6) on the *frozen* checkpoints.

Two studies, both scored under the identical Phase-3 protocol (``make_marl_env(region, "default")``,
disjoint eval seed stream ``seed + 100000 + episode``) and driven by the *same* action dispatcher
the frozen results and the Phase-5 GIFs use (``phase3_eval.select_actions``) — so every number here
is directly comparable to ``wildfire_phase3_multiseed/phase3_summary.json``.

1. **Transfer matrix** (train-on-A -> eval-on-B, both directions) for the learned policies that own
   region-specific checkpoints (CommNet, HierComm). Reports the full WEL/ISR matrix plus
   Transfer-Robustness-Score (ISR ratio), cross-domain WEL gap, and adaptation asymmetry.

2. **Held-out generalization / failure modes** per region under four stress conditions, each scored
   with the *falsifiable* ``dWEL = WEL(No-Op, condition) - WEL(policy, condition)`` computed WITHIN
   the condition (No-Op rerun per condition, so No-Op scores dWEL = 0 by construction):
     - ``standard``        : unmodified default regime (reference).
     - ``heldout_ignition``: ignition forced to fuel cells nearest the four map corners + center
                             (individually unlikely under the training ignition distribution).
     - ``wind_plus90``     : Weather.csv wind direction rotated +90 deg (all rows).
     - ``assets_rot90``    : asset_type/criticality/blast_radius rasters rotated 90 deg — an unseen
                             infrastructure layout over unchanged fuel/terrain/weather.
     - ``cross_region``    : the OTHER region's trained policy evaluated on this region.

Nothing in the RL/env/metric stack is modified — this is a read-only evaluation harness over the
frozen checkpoints.

Examples
--------
Smoke (validate end-to-end, ~minutes)::

    python scripts/run_phase6_transfer.py --ckpt-dir wildfire_phase3_multiseed \\
        --out results/phase6 --quick

Full (5 training seeds x 15 episodes, both directions, all conditions)::

    python scripts/run_phase6_transfer.py --ckpt-dir wildfire_phase3_multiseed \\
        --out results/phase6 --train-seeds 42,1042,2042,3042,4042 --episodes 15
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wildfire_marl.env.regimes import make_marl_env  # noqa: E402
from wildfire_marl.env.rewards import WELISRDeltaReward  # noqa: E402
from wildfire_marl.eval.metrics import (  # noqa: E402
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.eval.phase3_eval import (  # noqa: E402
    EVAL_OFFSET,
    LABELS,
    LEARNED,
    checkpoint_path,
    load_nets,
    select_actions,
)
from wildfire_marl.eval.significance import bootstrap_ci, welch_ttest  # noqa: E402
from wildfire_marl.eval.transfer import (  # noqa: E402
    adaptation_asymmetry,
    cross_domain_gap,
    transfer_robustness_score,
)

REGIONS = ["saudi", "california"]
MAPNAME = {"saudi": "Saudi", "california": "California"}
METRICS = ["WEL", "ISR", "CE", "burned", "return"]

#: Policies scored in both studies. Heuristics are env-driven (no checkpoint) so they behave
#: identically regardless of "training" region; only the learned policies genuinely transfer.
DEFAULT_POLICIES = ["noop", "value_first", "local_reactive", "commnet", "hiercomm_heur"]
#: Learned policies with region-specific checkpoints -> meaningful transfer matrix / cross-region.
TRANSFER_POLICIES = ["commnet", "hiercomm_heur"]
#: Proposed model (the shipped system).
PROPOSED = "hiercomm_heur"

ENVMODS = Path("results/phase6/envmods")

GEN_CONDITIONS = ["standard", "heldout_ignition", "wind_plus90", "assets_rot90", "cross_region"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- env construction


def _std_env(region: str, **overrides):
    """Default-regime env for ``region`` matching the frozen Phase-3 protocol exactly."""
    return make_marl_env(region, "default", reward_cls=WELISRDeltaReward, **overrides)


def _wind_env(region: str, delta: int = 90, **overrides):
    """Default env with Weather.csv wind direction rotated by ``delta`` degrees."""
    map_name = MAPNAME[region]
    parent = ENVMODS / f"wind{delta}"
    dst = parent / map_name
    if not dst.exists():
        shutil.copytree(Path("data/cell2fire") / map_name, dst)
        w = pd.read_csv(dst / "Weather.csv")
        w["WD"] = (w["WD"] + delta) % 360
        w.to_csv(dst / "Weather.csv", index=False)
    return make_marl_env(region, "default", data_dir=str(parent), reward_cls=WELISRDeltaReward,
                         **overrides)


def _rot_infra_env(region: str, **overrides):
    """Default env with asset_type/criticality/blast_radius rasters rotated 90 degrees."""
    map_name = MAPNAME[region]
    dst = ENVMODS / "infra_rot90" / map_name
    dst.mkdir(parents=True, exist_ok=True)
    src = Path("data/cell2fire") / map_name
    for layer in ("asset_type", "criticality", "blast_radius"):
        f = dst / f"{layer}.npy"
        if not f.exists():
            np.save(f, np.ascontiguousarray(np.rot90(np.load(src / f"{layer}.npy"))))
    return make_marl_env(region, "default", infra_dir=str(dst), reward_cls=WELISRDeltaReward,
                         **overrides)


def _stress_ignition_cells(env) -> list[int]:
    """Fuel cells nearest the four map corners + center (individually unlikely ignitions)."""
    fuel = env.env.fuel_mask
    h, w = fuel.shape
    anchors = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (h // 2, w // 2)]
    ys, xs = np.nonzero(fuel > 0)
    cells = []
    for ay, ax in anchors:
        d = np.abs(ys - ay) + np.abs(xs - ax)
        i = int(np.argmin(d))
        cells.append(int(ys[i] * w + xs[i]))
    return cells


# --------------------------------------------------------------------------- rollout


def rollout_once(env, kind, nets, device, seed, options=None) -> dict[str, float]:
    """One evaluation episode (optionally with fixed reset ``options``); WEL/ISR/CE/burned/return.

    Mirrors ``phase3_eval.rollout_episode`` but threads ``reset`` options (for the held-out
    ignition condition) — behaviour is otherwise byte-identical to the frozen protocol.
    """
    obs, info = env.reset(seed=seed, options=options)
    env.apply_target_compliance = False
    done, sc, ep_ret, ce = False, 0, 0.0, 1.0
    while not done:
        acts = select_actions(env, kind, nets, obs, info, device, sc)
        obs, rew, term, trunc, info = env.step(acts)
        ep_ret += float(rew["agent_0"])
        done = term["agent_0"] or trunc["agent_0"]
        ce = info[env.agents[0]].get("coordination_efficiency", ce)
        sc += 1
    fs = env.env.fire_state
    return {
        "WEL": float(weighted_economic_loss(env.env.asset_type, fs, env.env.asset_values)),
        "ISR": float(infrastructure_survival_rate(env.env.asset_type, fs)),
        "CE": float(ce),
        "burned": int((fs > 0).sum()),
        "return": ep_ret,
    }


def eval_policy_seedscores(
    env,
    kind: str,
    ckpt_region: str,
    ckpt_dir,
    device,
    eval_seeds: list[int],
    episodes: int,
    train_seeds: list[int],
    ignition_cells: list[int] | None = None,
) -> tuple[list[dict], dict[str, list[float]]]:
    """Evaluate ``kind`` (checkpoint from ``ckpt_region``) on ``env``.

    Learned policies contribute one score PER TRAINING SEED (mean over the eval stream) — the unit
    of statistical analysis, matching ``run_phase3``. Heuristics contribute a single score.
    Returns ``(raw_records, {metric: [per-seed means]})``.
    """
    raw: list[dict] = []
    per: dict[str, list[float]] = {m: [] for m in METRICS}
    seeds_to_use = train_seeds if kind in LEARNED else [None]
    for ts in seeds_to_use:
        nets = load_nets(kind, ckpt_region, ckpt_dir, env.num_agents, device, seed=ts)
        ep_metrics: dict[str, list[float]] = {m: [] for m in METRICS}
        for s in eval_seeds:
            for e in range(episodes):
                opts = None
                if ignition_cells is not None:
                    opts = {"ignition_cell": ignition_cells[e % len(ignition_cells)]}
                m = rollout_once(env, kind, nets, device, seed=s + EVAL_OFFSET + e, options=opts)
                m.update({"policy": kind, "ckpt_region": ckpt_region, "train_seed": ts,
                          "eval_seed": s, "episode": e})
                raw.append(m)
                for k in METRICS:
                    ep_metrics[k].append(m[k])
        for k in METRICS:
            per[k].append(float(np.mean(ep_metrics[k])))
    return raw, per


# --------------------------------------------------------------------------- transfer study


def run_transfer(ckpt_dir, out_dir: Path, policies, eval_seeds, episodes, train_seeds, device):
    """Full train-region x eval-region matrix for the learned transfer policies."""
    raw_all: list[dict] = []
    # cell[(policy, train_region, eval_region)] = {metric: [per-seed means]}
    cell: dict[tuple, dict] = {}
    for eval_region in REGIONS:
        env = _std_env(eval_region)
        for kind in policies:
            for train_region in REGIONS:
                raw, per = eval_policy_seedscores(
                    env, kind, train_region, ckpt_dir, device, eval_seeds, episodes, train_seeds
                )
                for r in raw:
                    r["eval_region"] = eval_region
                raw_all.extend(raw)
                cell[(kind, train_region, eval_region)] = per
                print(f"  [transfer] {kind}: train={train_region} eval={eval_region} "
                      f"WEL={np.mean(per['WEL']):.2f} ISR={np.mean(per['ISR']):.3f}", flush=True)
        env.close()

    summary: dict = {"protocol": {"train_seeds": train_seeds, "eval_seeds": eval_seeds,
                                  "episodes": episodes}, "policies": {}}
    for kind in policies:
        entry: dict = {"label": LABELS[kind], "directions": {}, "matrix": {}}
        for tr in REGIONS:
            for er in REGIONS:
                c = cell[(kind, tr, er)]
                entry["matrix"][f"{tr}->{er}"] = {
                    "WEL_mean": float(np.mean(c["WEL"])), "WEL_std": float(np.std(c["WEL"], ddof=1) if len(c["WEL"]) > 1 else 0.0),
                    "ISR_mean": float(np.mean(c["ISR"])), "ISR_std": float(np.std(c["ISR"], ddof=1) if len(c["ISR"]) > 1 else 0.0),
                }
        trs_pair = {}
        for a, b in (("saudi", "california"), ("california", "saudi")):
            nat = float(np.mean(cell[(kind, a, a)]["ISR"]))
            xfer = float(np.mean(cell[(kind, a, b)]["ISR"]))
            nat_wel = float(np.mean(cell[(kind, a, a)]["WEL"]))
            xfer_wel = float(np.mean(cell[(kind, a, b)]["WEL"]))
            trs = transfer_robustness_score(nat, xfer)
            trs_pair[f"{a}->{b}"] = trs
            # Welch on the per-seed ISR: native vs transferred (does transfer degrade?)
            w = (welch_ttest(cell[(kind, a, a)]["ISR"], cell[(kind, a, b)]["ISR"])
                 if len(train_seeds) > 1 else {"p_value": float("nan"), "cohens_d": float("nan")})
            entry["directions"][f"{a}->{b}"] = {
                "native_ISR": nat, "transfer_ISR": xfer, "TRS_ISR": trs,
                "native_WEL": nat_wel, "transfer_WEL": xfer_wel,
                "WEL_gap": cross_domain_gap(nat_wel, xfer_wel),
                "ISR_degradation_p": w["p_value"], "ISR_degradation_d": w["cohens_d"],
            }
        if all(k in trs_pair and not np.isnan(trs_pair[k]) for k in ("saudi->california", "california->saudi")):
            entry["adaptation_asymmetry_TRS"] = adaptation_asymmetry(
                trs_pair["saudi->california"], trs_pair["california->saudi"]
            )
        summary["policies"][kind] = entry

    pd.DataFrame(raw_all).to_csv(out_dir / "transfer_matrix_raw.csv", index=False)
    (out_dir / "transfer_summary.json").write_text(json.dumps(summary, indent=2))
    _emit_transfer_table(summary, policies, out_dir / "phase6_transfer_table.tex")
    return summary


def _emit_transfer_table(summary, policies, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{Cross-region transfer.} Native vs transferred ISR (higher is better) "
        r"with Transfer-Robustness-Score (TRS $=$ transfer/native ISR). Mean over training seeds.}",
        r"\label{tab:transfer}", r"\begin{tabular}{llccc}", r"\toprule",
        r"Policy & Direction & Native ISR & Transfer ISR & TRS \\", r"\midrule",
    ]
    for kind in policies:
        d = summary["policies"][kind]["directions"]
        for i, (dirn, v) in enumerate(d.items()):
            pol = LABELS[kind] if i == 0 else ""
            arrow = dirn.replace("saudi", "SA").replace("california", "CA").replace("->", r" $\to$ ")
            lines.append(rf"{pol} & {arrow} & {v['native_ISR']:.3f} & {v['transfer_ISR']:.3f} "
                         rf"& {v['TRS_ISR']:.2f} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- generalization study


def _condition_env(region: str, condition: str):
    """Build the env (and optional fixed-ignition cell list) for a stress condition."""
    if condition in ("standard", "cross_region", "heldout_ignition"):
        env = _std_env(region)
    elif condition == "wind_plus90":
        env = _wind_env(region, 90)
    elif condition == "assets_rot90":
        env = _rot_infra_env(region)
    else:
        raise ValueError(f"unknown condition {condition}")
    cells = _stress_ignition_cells(env) if condition == "heldout_ignition" else None
    return env, cells


def run_generalization(ckpt_dir, out_dir: Path, policies, eval_seeds, episodes, train_seeds, device):
    """Per-region stress conditions with falsifiable dWEL vs the per-condition No-Op baseline."""
    raw_all: list[dict] = []
    summary: dict = {"protocol": {"train_seeds": train_seeds, "eval_seeds": eval_seeds,
                                  "episodes": episodes}, "regions": {}}
    for region in REGIONS:
        other = "california" if region == "saudi" else "saudi"
        reg_summary: dict = {}
        for condition in GEN_CONDITIONS:
            env, cells = _condition_env(region, condition)
            # policies for this condition: cross_region uses the OTHER region's learned policies;
            # every other condition uses this region's own policies (+ heuristics as reference).
            if condition == "cross_region":
                cond_policies = [(k, other) for k in TRANSFER_POLICIES]
                cond_policies = [("noop", region)] + cond_policies  # No-Op baseline for dWEL
            else:
                cond_policies = [(k, region) for k in policies]

            wel_by_policy: dict[str, list[float]] = {}
            per_by_policy: dict[str, dict] = {}
            for kind, ckpt_region in cond_policies:
                raw, per = eval_policy_seedscores(
                    env, kind, ckpt_region, ckpt_dir, device, eval_seeds, episodes, train_seeds,
                    ignition_cells=cells,
                )
                for r in raw:
                    r["region"] = region
                    r["condition"] = condition
                raw_all.extend(raw)
                tag = kind if condition != "cross_region" or kind == "noop" else f"{kind}_from_{other}"
                wel_by_policy[tag] = per["WEL"]
                per_by_policy[tag] = per
                print(f"  [gen] {region}/{condition}: {tag} "
                      f"WEL={np.mean(per['WEL']):.2f} ISR={np.mean(per['ISR']):.3f}", flush=True)
            env.close()

            noop_wel = float(np.mean(wel_by_policy.get("noop", [float("nan")])))
            cond_entry = {}
            for tag, per in per_by_policy.items():
                wel_mean = float(np.mean(per["WEL"]))
                cond_entry[tag] = {
                    "WEL_mean": wel_mean,
                    "WEL_ci": list(bootstrap_ci(per["WEL"])) if len(per["WEL"]) > 1 else [wel_mean, wel_mean],
                    "ISR_mean": float(np.mean(per["ISR"])),
                    "dWEL": noop_wel - wel_mean,  # >0 == policy saves value vs No-Op in this condition
                    "n_seeds": len(per["WEL"]),
                }
            reg_summary[condition] = cond_entry
        summary["regions"][region] = reg_summary

    pd.DataFrame(raw_all).to_csv(out_dir / "generalization_raw.csv", index=False)
    (out_dir / "generalization_summary.json").write_text(json.dumps(summary, indent=2))
    _emit_generalization_table(summary, out_dir / "phase6_generalization_table.tex")
    return summary


def _emit_generalization_table(summary, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{Generalization / failure modes.} $\Delta$WEL $=$ WEL(No-Op) $-$ "
        r"WEL(policy) within each condition (higher is better; No-Op $=0$ by construction).}",
        r"\label{tab:generalization}", r"\begin{tabular}{llc}", r"\toprule",
        r"Region & Condition & $\Delta$WEL (proposed) \\", r"\midrule",
    ]
    for region, conds in summary["regions"].items():
        rlabel = {"saudi": "Saudi", "california": "California"}[region]
        for i, (cond, entry) in enumerate(conds.items()):
            tag = PROPOSED if PROPOSED in entry else next(
                (t for t in entry if t.startswith(PROPOSED)), None)
            dwel = f"{entry[tag]['dWEL']:.2f}" if tag else "--"
            lines.append(rf"{rlabel if i == 0 else ''} & {cond.replace('_', chr(92)+'_')} & {dwel} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-6 transfer + generalization (frozen checkpoints).")
    ap.add_argument("--ckpt-dir", default="wildfire_phase3_multiseed")
    ap.add_argument("--out", default="results/phase6")
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    ap.add_argument("--eval-seeds", default="42,1042,2042,3042,4042")
    ap.add_argument("--train-seeds", default="42,1042,2042,3042,4042")
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--study", default="both", choices=["transfer", "generalization", "both"])
    ap.add_argument("--quick", action="store_true", help="1 train seed x 1 eval seed x 1 episode")
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    eval_seeds = [int(s) for s in args.eval_seeds.split(",") if s.strip()]
    train_seeds = [int(s) for s in args.train_seeds.split(",") if s.strip()]
    episodes = args.episodes
    if args.quick:
        eval_seeds, train_seeds, episodes = eval_seeds[:1], train_seeds[:1], 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ENVMODS.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    device = torch.device("cpu")

    print(f"Phase-6 | ckpt={args.ckpt_dir} study={args.study} policies={policies}\n"
          f"         train_seeds={train_seeds} eval_seeds={eval_seeds} episodes={episodes}",
          flush=True)

    if args.study in ("transfer", "both"):
        print("\n=== TRANSFER MATRIX ===", flush=True)
        transfer_policies = [p for p in policies if p in TRANSFER_POLICIES]
        run_transfer(args.ckpt_dir, out_dir, transfer_policies, eval_seeds, episodes, train_seeds, device)

    if args.study in ("generalization", "both"):
        print("\n=== GENERALIZATION / FAILURE MODES ===", flush=True)
        run_generalization(args.ckpt_dir, out_dir, policies, eval_seeds, episodes, train_seeds, device)

    print(f"\nPhase-6 complete. Artifacts -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
