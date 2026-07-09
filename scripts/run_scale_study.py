"""Phase 5 (peer-review remediation): agent-count scale study (M6).

N in {3, 6, 10} suppression agents on the 32x32 regions, certified protocol
(5 seeds x 15 episodes). Policies: No-Op, Value-First Heuristic, Local Reactive, and the
N=3-trained CommNet evaluated ZERO-SHOT at larger team sizes (its mean-pooled
communication handles arbitrary N). The learned hierarchy is architecturally fixed to
N=3 (per-agent commander heads) and is excluded above N=3 — reported as a scaling
limitation. The 64x64 grid axis is documented as infeasible this pass: the Cell2Fire
landscape conversion and the infrastructure rasters exist only at 32x32
(data/*/grids/64x64/ has no infrastructure layers), and all policies would require
retraining.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from scripts.run_multiseed_eval_v2 import BehaviorLog, bootstrap_ci, run_episode

from wildfire_marl.agents.agent_networks import CommNetActor
from wildfire_marl.env.marl_env import MultiAgentFireEnv

POLICIES = ["No-Op", "Value-First Heuristic", "Local Reactive", "CommNet (matched)"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Phase 5 agent-count scale study.")
    parser.add_argument("--region", type=str, required=True, choices=["saudi", "california"])
    parser.add_argument("--agents", type=str, default="3,6,10")
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/phase5_peer")
    args = parser.parse_args()

    set_seed(args.base_seed)
    device = torch.device("cpu")
    region = args.region
    map_name = "Saudi" if region == "saudi" else "California"
    seeds = [args.base_seed + s * 1000 for s in range(args.seeds)]
    agent_counts = [int(x) for x in args.agents.split(",")]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(f"results/phase2_peer/checkpoint_commnet_{region}.pt", map_location=device)
    commnet = CommNetActor(in_channels=8, action_dim=6, features_dim=64, comm_rounds=2).to(device)
    commnet.load_state_dict(ck["actor_state_dict"])
    commnet.eval()
    nets = {"commnet": commnet}

    records = []
    for n in agent_counts:
        env = MultiAgentFireEnv(
            num_agents=n,
            crop_size=9,
            coordination_penalty=0.1,
            fire_map=map_name,
            data_dir="data/cell2fire",
            max_steps=150,
            steps_per_action=60,
            observe_infra=True,
            catastrophe_weight=2.0,
            cascade_prob=0.1,
            infra_dir=f"data/cell2fire/{map_name}",
        )
        for policy in POLICIES:
            behavior = BehaviorLog(env.height, env.width, env.env.asset_type)
            print(f"[{region}|N={n}] {policy}", flush=True)
            for seed in seeds:
                for ep in range(args.episodes):
                    m = run_episode(env, policy, nets, seed + ep, device, behavior)
                    records.append(
                        {"region": region, "num_agents": n, "policy": policy,
                         "seed": seed, "episode": ep} | m
                    )
        env.close()

    raw = pd.DataFrame(records)
    raw.to_csv(out / f"scale_raw_{region}.csv", index=False)
    agg = raw.groupby(["region", "num_agents", "policy", "seed"], as_index=False).mean(
        numeric_only=True
    )
    agg.to_csv(out / f"scale_aggregate_{region}.csv", index=False)

    summary: dict = {
        "region": region,
        "protocol": {"seeds": seeds, "episodes_per_seed": args.episodes, "unit": "seed means"},
        "note": "CommNet is the N=3-trained checkpoint evaluated zero-shot at larger N",
        "by_num_agents": {},
    }
    for n in agent_counts:
        sub = agg[agg.num_agents == n]
        summary["by_num_agents"][str(n)] = {}
        for policy in POLICIES:
            v = sub.loc[sub.policy == policy, "WEL"].to_numpy()
            lo, hi = bootstrap_ci(v)
            isr = sub.loc[sub.policy == policy, "ISR"].to_numpy()
            summary["by_num_agents"][str(n)][policy] = {
                "WEL_mean": float(v.mean()),
                "WEL_ci": [lo, hi],
                "ISR_mean": float(isr.mean()),
            }
    with open(out / f"scale_summary_{region}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{region}] scale study complete -> {out}")


if __name__ == "__main__":
    main()
