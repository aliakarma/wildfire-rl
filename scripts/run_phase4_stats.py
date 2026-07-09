"""Phase 4 (peer-review remediation): statistical reanalysis & claims recalibration.

Produces, under results/phase4_peer/:
- compliance_reassignment_{region}.json : Table 2 metrics recomputed under the certified
  protocol and ARCHIVED (audit issue M12), plus the commander reassignment rates that
  answer Reviewer #1's Question 2 (M2).
- tost_results.json : TOST equivalence tests (H7) with a pre-registered +/-5% margin —
  Hierarchical vs Value-First (both regions) and BC-only vs Full System (Saudi).
- stats_tables.json : uniform Welch tests (t, Welch-Satterthwaite df, p), pooled-SD
  Cohen's d with degenerate-variance flags, mean differences with bootstrap CIs — for all
  Phase 2 policies vs No-Op / vs Hierarchical, both regions, and Phase 3 ablations vs the
  full system (M8, C2 support).
- reward_decomposition.json : constant vs policy-dependent components of the total_reward
  column (L1).

Statistical unit everywhere: 5 seed means (n = 5 per policy).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import random

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps
from scripts.eval_hierarchical import target_seeking_action

from wildfire_marl.agents.strategic_controller import StrategicController, get_sector_center
from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.eval.compliance import analyze_compliance_trajectory
from wildfire_marl.train.hierarchical_train import extract_high_level_state

OUT = Path("results/phase4_peer")
EQUIV_MARGIN_FRAC = 0.05  # pre-registered: +/-5% of the reference policy's mean WEL


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sector_of(pos: tuple[int, int]) -> int:
    return int(np.clip(pos[0] // 8, 0, 3)) * 4 + int(np.clip(pos[1] // 8, 0, 3))


# ---------------------------------------------------------------- part A: instrumented run
def instrumented_hierarchical_run(region: str, episodes: int, seeds: list[int]) -> dict:
    device = torch.device("cpu")
    map_name = "Saudi" if region == "saudi" else "California"
    data_dir = "data/cell2fire"
    env = MultiAgentFireEnv(
        num_agents=3,
        crop_size=9,
        coordination_penalty=0.1,
        fire_map=map_name,
        data_dir=data_dir,
        max_steps=150,
        steps_per_action=60,
        observe_infra=True,
        catastrophe_weight=2.0,
        cascade_prob=0.1,
        infra_dir=f"{data_dir}/{map_name}",
    )
    ck = torch.load(f"results/runs/checkpoint_hierarchical_{region}.pt", map_location=device)
    commander = StrategicController(num_agents=env.num_agents).to(device)
    commander.load_state_dict(ck["commander_state_dict"])
    commander.eval()

    comp_rates, drifts, latencies = [], [], []
    cross_sector_dispatches = 0  # assigned sector != agent's sector at dispatch
    assignment_changes = 0  # assignment != previous assignment
    dispatch_events = 0
    change_events = 0

    for seed in seeds:
        for ep in range(episodes):
            obs_dict, info_dict = env.reset(seed=seed + ep)
            env.apply_target_compliance = False
            pos_hist = {a: [env.agent_positions[a]] for a in env.agents}
            tar_hist = {a: [env.agent_positions[a]] for a in env.agents}
            curr_targets = {a: env.agent_positions[a] for a in env.agents}
            prev_assign: dict[str, int] | None = None
            done = False
            step = 0
            while not done:
                if step % 10 == 0:
                    s_fire, s_asset, s_agent = extract_high_level_state(env)
                    with torch.no_grad():
                        logits = commander(s_fire, s_asset, s_agent)
                        assign = {
                            a: int(torch.argmax(logits[i]))
                            for i, a in enumerate(env.agents)
                        }
                    for a in env.agents:
                        curr_targets[a] = get_sector_center(assign[a])
                        dispatch_events += 1
                        if assign[a] != sector_of(env.agent_positions[a]):
                            cross_sector_dispatches += 1
                        if prev_assign is not None:
                            change_events += 1
                            if assign[a] != prev_assign[a]:
                                assignment_changes += 1
                    prev_assign = assign
                env.strategic_targets = curr_targets
                actions = {
                    a: target_seeking_action(env, a, info_dict[a]["action_mask"])
                    for a in env.agents
                }
                obs_dict, _, term, trunc, info_dict = env.step(actions)
                done = term["agent_0"] or trunc["agent_0"]
                for a in env.agents:
                    pos_hist[a].append(env.agent_positions[a])
                    tar_hist[a].append(env.strategic_targets[a])
                step += 1

            rep = analyze_compliance_trajectory(pos_hist, tar_hist)
            comp_rates.append(rep["target_compliance_rate"])
            drifts.append(rep["sector_drift"])
            latencies.append(rep["dispatch_latency"])

    env.close()
    return {
        "region": region,
        "protocol": {"seeds": seeds, "episodes_per_seed": episodes, "n_episodes": len(comp_rates)},
        "table2_metrics": {
            "target_compliance_rate": {
                "mean": float(np.mean(comp_rates)),
                "std": float(np.std(comp_rates, ddof=1)),
            },
            "sector_drift_cells": {
                "mean": float(np.mean(drifts)),
                "std": float(np.std(drifts, ddof=1)),
            },
            "dispatch_latency_steps": {
                "mean": float(np.mean(latencies)),
                "std": float(np.std(latencies, ddof=1)),
            },
        },
        "reassignment": {
            "cross_sector_dispatch_rate": cross_sector_dispatches / max(dispatch_events, 1),
            "assignment_change_rate": assignment_changes / max(change_events, 1),
            "dispatch_events": dispatch_events,
            "change_events": change_events,
        },
    }


# ---------------------------------------------------------------- part B: test machinery
def welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch test of a vs b with Welch-Satterthwaite df; diff = mean(a) - mean(b)."""
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    diff = float(a.mean() - b.mean())
    out: dict = {"n_a": na, "n_b": nb, "mean_diff": diff}
    if va + vb == 0:
        out |= {"t": None, "df": None, "p": None, "cohens_d": None, "degenerate": "both"}
        return out
    se = np.sqrt(va / na + vb / nb)
    t = diff / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    p = 2 * sps.t.sf(abs(t), df)
    pooled = np.sqrt((va + vb) / 2)
    out |= {
        "t": float(t),
        "df": float(df),
        "p": float(p),
        "cohens_d": float(diff / pooled),
        "degenerate": ("a" if va == 0 else "b" if vb == 0 else None),
    }
    # bootstrap CI of the mean difference
    rng = np.random.default_rng(0)
    boots = rng.choice(a, (10000, na)).mean(1) - rng.choice(b, (10000, nb)).mean(1)
    out["diff_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def tost(a: np.ndarray, b: np.ndarray, margin: float) -> dict:
    """Two one-sided Welch tests for equivalence of mean(a) and mean(b) within +/-margin."""
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    diff = float(a.mean() - b.mean())
    t_lower = (diff + margin) / se  # H1: diff > -margin
    t_upper = (diff - margin) / se  # H1: diff < +margin
    p_lower = float(sps.t.sf(t_lower, df))
    p_upper = float(sps.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)
    return {
        "mean_diff": diff,
        "margin": margin,
        "se": float(se),
        "df": float(df),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent_at_0.05": bool(p_tost < 0.05),
    }


def seed_means(df: pd.DataFrame, policy: str, metric: str = "WEL") -> np.ndarray:
    return df.loc[df.policy == policy, metric].to_numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    # ---- Part A: compliance + reassignment (certified protocol, shipped checkpoint)
    seeds = [42 + s * 1000 for s in range(5)]
    compliance = {}
    for region in ["saudi", "california"]:
        print(f"[{region}] instrumented hierarchical run...", flush=True)
        rep = instrumented_hierarchical_run(region, episodes=15, seeds=seeds)
        compliance[region] = rep
        with open(OUT / f"compliance_reassignment_{region}.json", "w") as f:
            json.dump(rep, f, indent=2)

    # ---- Load aggregates (Phase 2 + reactive + Phase 3)
    agg = {}
    for region in ["saudi", "california"]:
        main_a = pd.read_csv(f"results/phase2_peer/multiseed_eval_aggregate_{region}.csv")
        react = pd.read_csv(
            f"results/phase2_peer/multiseed_eval_aggregate_{region}_reactive.csv"
        )
        agg[region] = pd.concat([main_a, react], ignore_index=True)
    abl = pd.read_csv("results/phase3_peer/ablation_aggregate_saudi.csv")

    # ---- Part B: TOST equivalence (pre-registered margin: 5% of reference mean WEL)
    tost_out = {"margin_definition": f"+/-{EQUIV_MARGIN_FRAC:.0%} of reference policy mean WEL"}
    for region in ["saudi", "california"]:
        h = seed_means(agg[region], "Learned Hierarchical")
        vf = seed_means(agg[region], "Value-First Heuristic")
        tost_out[f"{region}: Hierarchical vs Value-First (ref)"] = tost(
            h, vf, EQUIV_MARGIN_FRAC * vf.mean()
        )
    bc = seed_means(abl, "BC-only (no RL fine-tuning)")
    full = seed_means(abl, "Full System (shipped ckpt)")
    tost_out["saudi: BC-only vs Full System (ref)"] = tost(
        bc, full, EQUIV_MARGIN_FRAC * full.mean()
    )
    with open(OUT / "tost_results.json", "w") as f:
        json.dump(tost_out, f, indent=2)

    # ---- Part C: uniform Welch tables (WEL and ISR), sign convention documented
    tables: dict = {
        "convention": "mean_diff = policy - reference; WEL negative = policy better; "
        "unit = seed means (n=5); Welch t with Welch-Satterthwaite df; "
        "Cohen's d pooled-SD; d suppressed when either side degenerate",
    }
    for region in ["saudi", "california"]:
        tables[region] = {}
        df_r = agg[region]
        for metric in ["WEL", "ISR"]:
            tables[region][metric] = {}
            noop = seed_means(df_r, "No-Op", metric)
            hier = seed_means(df_r, "Learned Hierarchical", metric)
            for pol in df_r.policy.unique():
                res = {}
                if pol != "No-Op":
                    res["vs_NoOp"] = welch(seed_means(df_r, pol, metric), noop)
                if pol != "Learned Hierarchical":
                    res["vs_Hierarchical"] = welch(seed_means(df_r, pol, metric), hier)
                tables[region][metric][pol] = res
    tables["saudi_ablations"] = {}
    full_w = seed_means(abl, "Full System (shipped ckpt)")
    for pol in abl.policy.unique():
        if pol != "Full System (shipped ckpt)":
            tables["saudi_ablations"][pol] = {
                "vs_FullShipped": welch(seed_means(abl, pol), full_w)
            }
    with open(OUT / "stats_tables.json", "w") as f:
        json.dump(tables, f, indent=2)

    # ---- Part D: reward decomposition (L1)
    decomp = {
        "note": "total_reward is dominated by a policy-independent burn constant; "
        "policy effect = delta vs No-Op seed means"
    }
    for region in ["saudi", "california"]:
        df_r = agg[region]
        noop_r = seed_means(df_r, "No-Op", "total_reward")
        decomp[region] = {"noop_baseline_mean": float(noop_r.mean())}
        for pol in df_r.policy.unique():
            if pol == "No-Op":
                continue
            v = seed_means(df_r, pol, "total_reward")
            delta = welch(v, noop_r)
            decomp[region][pol] = {
                "delta_vs_noop": delta["mean_diff"],
                "delta_ci95": delta.get("diff_ci95"),
                "delta_as_fraction_of_baseline": abs(delta["mean_diff"])
                / abs(float(noop_r.mean())),
            }
    with open(OUT / "reward_decomposition.json", "w") as f:
        json.dump(decomp, f, indent=2)

    # ---- console summary
    print("\n===== TABLE 2 (recomputed, archived) + REASSIGNMENT =====")
    for region, rep in compliance.items():
        t2 = rep["table2_metrics"]
        ra = rep["reassignment"]
        print(
            f"  {region:11s} compliance {t2['target_compliance_rate']['mean']:.3f} | "
            f"drift {t2['sector_drift_cells']['mean']:.2f} | "
            f"latency {t2['dispatch_latency_steps']['mean']:.2f} | "
            f"cross-sector dispatch {ra['cross_sector_dispatch_rate']:.1%} | "
            f"assignment change {ra['assignment_change_rate']:.1%}"
        )
    print("\n===== TOST =====")
    for k, v in tost_out.items():
        if isinstance(v, dict):
            print(
                f"  {k}: diff {v['mean_diff']:+.3f} margin ±{v['margin']:.3f} "
                f"p_tost {v['p_tost']:.4f} equivalent: {v['equivalent_at_0.05']}"
            )
    print("\nDone ->", OUT)


if __name__ == "__main__":
    main()
