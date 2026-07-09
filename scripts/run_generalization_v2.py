"""Phase 5 (peer-review remediation): held-out generalization suite, replacing TRS (M1/H9).

Conditions per region, each evaluated under the certified protocol (5 seeds x 15 episodes):
- heldout_ignition : stress ignition cells — the 4 fuel cells nearest the map corners plus
  the one nearest the center, cycled by episode index (individually unlikely under the
  training-time uniform-fuel-cell ignition distribution).
- wind_plus90      : Weather.csv wind direction rotated +90 degrees (all rows).
- assets_rot90     : asset_type/criticality/blast_radius rasters rotated 90 degrees —
  unseen infrastructure layout over unchanged fuel/terrain/weather physics.
- cross_region     : the OTHER region's trained policies evaluated on this region's
  standard environment (the corrected version of the paper's broken transfer study).

Metric: WEL/ISR per policy per condition, plus dWEL = WEL(No-Op) - WEL(policy) computed
WITHIN each condition (No-Op is rerun per condition). No-Op scores dWEL = 0 by
construction — the falsifiability property the old TRS lacked.

Policies: No-Op, Value-First Heuristic, Local Reactive, Learned Hierarchical (shipped),
CommNet (matched). Cross-region uses the two trained policies only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
import shutil

import numpy as np
import pandas as pd
import torch
from scripts.run_multiseed_eval_v2 import BehaviorLog, bootstrap_ci, run_episode, welch_and_d

from wildfire_marl.agents.agent_networks import CommNetActor
from wildfire_marl.agents.strategic_controller import StrategicController
from wildfire_marl.env.marl_env import MultiAgentFireEnv

ENVMODS = Path("results/phase5_peer/envmods")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(map_name: str, data_dir: str, infra_dir: str) -> MultiAgentFireEnv:
    return MultiAgentFireEnv(
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
        infra_dir=infra_dir,
    )


def prepare_wind_shift(map_name: str, delta_deg: int) -> tuple[str, str]:
    """Copy the region landscape dir with WD rotated by delta_deg; return (data_dir, map)."""
    src = Path("data/cell2fire") / map_name
    dst_name = f"{map_name}_wind{delta_deg}"
    dst = ENVMODS / dst_name
    if not dst.exists():
        shutil.copytree(src, dst)
        w = pd.read_csv(dst / "Weather.csv")
        w["WD"] = (w["WD"] + delta_deg) % 360
        w.to_csv(dst / "Weather.csv", index=False)
    return str(ENVMODS), dst_name


def prepare_rotated_infra(map_name: str) -> str:
    """Copy infra rasters rotated 90 degrees; return the new infra_dir."""
    src = Path("data/cell2fire") / map_name
    dst = ENVMODS / f"{map_name}_infra_rot90"
    dst.mkdir(parents=True, exist_ok=True)
    for layer in ["asset_type", "criticality", "blast_radius"]:
        f = dst / f"{layer}.npy"
        if not f.exists():
            np.save(f, np.ascontiguousarray(np.rot90(np.load(src / f"{layer}.npy"))))
    return str(dst)


def stress_ignitions(env: MultiAgentFireEnv, k_center: bool = True) -> list[int]:
    """Fuel cells nearest the 4 corners (+ center): individually unlikely ignitions."""
    fuel = env.env.fuel_mask
    h, w = fuel.shape
    anchors = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    if k_center:
        anchors.append((h // 2, w // 2))
    ys, xs = np.nonzero(fuel > 0)
    cells = []
    for ay, ax in anchors:
        d = np.abs(ys - ay) + np.abs(xs - ax)
        i = int(np.argmin(d))
        cells.append(int(ys[i] * w + xs[i]))
    return cells


def load_nets(region: str, device) -> dict:
    nets: dict = {}
    ck = torch.load(f"results/runs/checkpoint_hierarchical_{region}.pt", map_location=device)
    commander = StrategicController(num_agents=3).to(device)
    commander.load_state_dict(ck["commander_state_dict"])
    commander.eval()
    nets["commander"] = commander
    ck = torch.load(f"results/phase2_peer/checkpoint_commnet_{region}.pt", map_location=device)
    commnet = CommNetActor(in_channels=8, action_dim=6, features_dim=64, comm_rounds=2).to(device)
    commnet.load_state_dict(ck["actor_state_dict"])
    commnet.eval()
    nets["commnet"] = commnet
    return nets


HELD_OUT_POLICIES = [
    "No-Op",
    "Value-First Heuristic",
    "Local Reactive",
    "Learned Hierarchical",
    "CommNet (matched)",
]
CROSS_POLICIES = ["No-Op", "Learned Hierarchical", "CommNet (matched)"]


def evaluate(
    env: MultiAgentFireEnv,
    policies: list[str],
    nets: dict,
    seeds: list[int],
    episodes: int,
    region: str,
    condition: str,
    ignition_cells: list[int] | None = None,
) -> list[dict]:
    device = torch.device("cpu")
    records = []
    for policy in policies:
        behavior = BehaviorLog(env.height, env.width, env.env.asset_type)
        print(f"[{region}|{condition}] {policy}", flush=True)
        for seed in seeds:
            for ep in range(episodes):
                opts = None
                if ignition_cells is not None:
                    opts = {"ignition_cell": ignition_cells[ep % len(ignition_cells)]}
                m = run_episode(
                    env, policy, nets, seed + ep, device, behavior, reset_options=opts
                )
                records.append(
                    {
                        "region": region,
                        "condition": condition,
                        "policy": policy,
                        "seed": seed,
                        "episode": ep,
                    }
                    | m
                )
    return records


def main():
    parser = argparse.ArgumentParser(description="Phase 5 held-out generalization suite.")
    parser.add_argument("--region", type=str, required=True, choices=["saudi", "california"])
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/phase5_peer")
    args = parser.parse_args()

    set_seed(args.base_seed)
    device = torch.device("cpu")
    region = args.region
    other = "california" if region == "saudi" else "saudi"
    map_name = "Saudi" if region == "saudi" else "California"
    seeds = [args.base_seed + s * 1000 for s in range(args.seeds)]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ENVMODS.mkdir(parents=True, exist_ok=True)

    nets_native = load_nets(region, device)
    nets_other = load_nets(other, device)

    records: list[dict] = []

    # -- condition 1: held-out stress ignitions (standard env)
    env = make_env(map_name, "data/cell2fire", f"data/cell2fire/{map_name}")
    cells = stress_ignitions(env)
    print(f"[{region}] stress ignition cells: {cells}")
    records += evaluate(
        env, HELD_OUT_POLICIES, nets_native, seeds, args.episodes, region,
        "heldout_ignition", ignition_cells=cells,
    )

    # -- condition 4 (same env): corrected cross-region transfer (other region's nets)
    cross = evaluate(
        env, CROSS_POLICIES, nets_other, seeds, args.episodes, region, "cross_region"
    )
    for r in cross:
        if r["policy"] != "No-Op":
            r["policy"] = f"{r['policy']} [trained on {other}]"
    records += cross
    env.close()

    # -- condition 2: wind +90 degrees
    dd, mn = prepare_wind_shift(map_name, 90)
    env = make_env(mn, dd, f"data/cell2fire/{map_name}")
    records += evaluate(
        env, HELD_OUT_POLICIES, nets_native, seeds, args.episodes, region, "wind_plus90"
    )
    env.close()

    # -- condition 3: rotated asset layout
    infra = prepare_rotated_infra(map_name)
    env = make_env(map_name, "data/cell2fire", infra)
    records += evaluate(
        env, HELD_OUT_POLICIES, nets_native, seeds, args.episodes, region, "assets_rot90"
    )
    env.close()

    raw = pd.DataFrame(records)
    raw.to_csv(out / f"generalization_raw_{region}.csv", index=False)
    agg = raw.groupby(["region", "condition", "policy", "seed"], as_index=False).mean(
        numeric_only=True
    )
    agg.to_csv(out / f"generalization_aggregate_{region}.csv", index=False)

    # summary: per condition, dWEL vs the SAME-condition No-Op (falsifiable by design)
    summary: dict = {
        "region": region,
        "protocol": {"seeds": seeds, "episodes_per_seed": args.episodes, "unit": "seed means"},
        "metric": "dWEL = mean WEL(No-Op, same condition) - mean WEL(policy); "
        ">0 = policy adds value; No-Op = 0 by construction",
        "conditions": {},
    }
    for cond in agg.condition.unique():
        sub = agg[agg.condition == cond]
        noop = sub.loc[sub.policy == "No-Op", "WEL"].to_numpy()
        summary["conditions"][cond] = {}
        for policy in sub.policy.unique():
            v = sub.loc[sub.policy == policy, "WEL"].to_numpy()
            lo, hi = bootstrap_ci(v)
            entry = {
                "WEL_mean": float(v.mean()),
                "WEL_ci": [lo, hi],
                "per_seed": [round(float(x), 3) for x in v],
                "dWEL_vs_noop": float(noop.mean() - v.mean()),
            }
            if policy != "No-Op":
                entry["welch_vs_noop"] = welch_and_d(v, noop)
            isr = sub.loc[sub.policy == policy, "ISR"].to_numpy()
            entry["ISR_mean"] = float(isr.mean())
            summary["conditions"][cond][policy] = entry

    with open(out / f"generalization_summary_{region}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{region}] generalization suite complete -> {out}")


if __name__ == "__main__":
    main()
