"""Phase 3 (peer-review remediation) ablation study under the certified protocol.

Fixes audit issues C1/H2/H8:
- Every ablation variant is trained once (seed 42, the shipped checkpoint's embedded
  hyperparameters) and evaluated under the certified protocol: 5 seeds x 15 episodes,
  identical seed derivation to phase8/phase2 (seeds 42+1000s, ep_seed = seed + ep).
- The "Full System (shipped)" row is NOT retrained: it reuses the Phase 2 certified
  evaluation rows of results/runs/checkpoint_hierarchical_{region}.pt, making it identical
  to Table 1's Hierarchical row by construction (C1).
- A "Full System (retrained)" control row quantifies training-run variance — the confound
  behind the old Table 3 (17.4 vs 20.28).
- A "BC-only (no RL fine-tuning)" row isolates what KL-regularized REINFORCE buys (H2).
- The "w/o Target-Seeking" variant uses the Phase 2 matched MAPPO actor (100k steps,
  WEL/ISR objective) as the frozen tactical layer — a fair version of the original ablation.
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
from scripts.eval_hierarchical import target_seeking_action
from scripts.run_multiseed_eval_v2 import bootstrap_ci, welch_and_d

from wildfire_marl.agents.agent_networks import MAPPOActor
from wildfire_marl.agents.strategic_controller import StrategicController, get_sector_center
from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.eval.metrics import (
    burned_cells,
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.train.hierarchical_train import extract_high_level_state, train_hierarchical

# The shipped checkpoint's embedded config (verified in Phase 0 of peer_phases.md)
BASE_CFG = {
    "pretrain_episodes": 40,
    "bc_epochs": 100,
    "hierarchical_episodes": 120,
    "kl_coef": 0.3,
    "ent_coef": 0.05,
    "isr_weight": 20.0,
    "gamma": 0.99,
    "rl_lr": 5e-5,
}

VARIANTS: dict[str, dict] = {
    "Full System (retrained)": {},
    "BC-only (no RL fine-tuning)": {"hierarchical_episodes": 0},
    "w/o BC Pretraining": {"pretrain_episodes": 0},
    "w/o KL Regularization": {"kl_coef": 0.0},
    "w/o Entropy Bonus": {"ent_coef": 0.0},
    "w/o WEL/ISR Reward": {"use_raw_reward": True},
    "w/o Target-Seeking": {"use_mappo_tactical": True},
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def eval_episode(
    env: MultiAgentFireEnv,
    commander: StrategicController,
    tactical: str,
    mappo_actor: MAPPOActor | None,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """One certified-protocol episode with the given commander and tactical layer."""
    obs_dict, info_dict = env.reset(seed=seed)
    env.apply_target_compliance = False
    curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}
    done = False
    step_count = 0
    ep_reward = 0.0

    while not done:
        if step_count % 10 == 0:
            s_fire, s_asset, s_agent = extract_high_level_state(env)
            with torch.no_grad():
                logits_list = commander(
                    s_fire.to(device), s_asset.to(device), s_agent.to(device)
                )
                actions = [torch.argmax(logits).item() for logits in logits_list]
            for idx, agent in enumerate(env.agents):
                curr_targets[agent] = get_sector_center(actions[idx])

        env.strategic_targets = curr_targets

        actions_dict = {}
        for agent in env.agents:
            mask = info_dict[agent]["action_mask"]
            if tactical == "mappo":
                with torch.no_grad():
                    obs_t = torch.tensor(
                        obs_dict[agent], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                    logits = mappo_actor(obs_t, mask_t).squeeze(0)
                actions_dict[agent] = int(torch.argmax(logits).item())
            else:
                actions_dict[agent] = target_seeking_action(env, agent, mask)

        obs_dict, rewards_dict, terminations_dict, truncations_dict, info_dict = env.step(
            actions_dict
        )
        ep_reward += rewards_dict["agent_0"]
        done = terminations_dict["agent_0"] or truncations_dict["agent_0"]
        step_count += 1

    final_state = env.env.fire_state
    return {
        "WEL": weighted_economic_loss(
            asset_type=env.env.asset_type,
            final_state=final_state,
            asset_values=env.env.asset_values,
        ),
        "ISR": infrastructure_survival_rate(
            asset_type=env.env.asset_type, final_state=final_state
        ),
        "CE": info_dict[env.agents[0]].get("coordination_efficiency", 1.0),
        "burned_cells": burned_cells(final_state),
        "total_reward": ep_reward,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 ablation study (certified protocol).")
    parser.add_argument("--region", type=str, default="saudi", choices=["saudi", "california"])
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument(
        "--mappo_ckpt_dir",
        type=str,
        default="results/phase2_peer",
        help="matched MAPPO actor for the w/o Target-Seeking variant",
    )
    parser.add_argument("--output_dir", type=str, default="results/phase3_peer")
    parser.add_argument(
        "--phase2_aggregate",
        type=str,
        default="results/phase2_peer/multiseed_eval_aggregate_{region}.csv",
        help="source of the Full System (shipped) rows — same runs as Table 1",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="tiny training budgets for a pipeline test"
    )
    args = parser.parse_args()

    if args.smoke:
        BASE_CFG.update({"pretrain_episodes": 2, "bc_epochs": 5, "hierarchical_episodes": 2})

    device = torch.device("cpu")
    map_name = "Saudi" if args.region.lower() == "saudi" else "California"
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

    mappo_actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
    ck = torch.load(
        Path(args.mappo_ckpt_dir) / f"checkpoint_mappo_{args.region}.pt", map_location=device
    )
    mappo_actor.load_state_dict(ck["actor_state_dict"])
    mappo_actor.eval()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_seeds = [args.base_seed + s * 1000 for s in range(args.seeds)]

    records = []
    for name, overrides in VARIANTS.items():
        cfg = BASE_CFG | overrides | {"seed": args.train_seed, "region": args.region}
        tag = (
            name.lower()
            .replace(" ", "_")
            .replace("/", "")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )
        print(f"\n===== TRAIN {name} =====", flush=True)
        set_seed(args.train_seed)
        commander = train_hierarchical(env, mappo_actor, cfg, device)
        commander.eval()
        torch.save(
            {"commander_state_dict": commander.state_dict(), "config": cfg},
            out / f"checkpoint_ablation_{tag}_{args.region}.pt",
        )

        tactical = "mappo" if overrides.get("use_mappo_tactical") else "target_seeking"
        print(f"===== EVAL {name} (tactical={tactical}) =====", flush=True)
        for seed in eval_seeds:
            for ep in range(args.episodes):
                m = eval_episode(env, commander, tactical, mappo_actor, seed + ep, device)
                records.append(
                    {
                        "seed": seed,
                        "episode": ep,
                        "region": args.region,
                        "policy": name,
                    }
                    | m
                )

    raw = pd.DataFrame(records)
    raw.to_csv(out / f"ablation_raw_{args.region}.csv", index=False)

    agg = raw.groupby(["region", "policy", "seed"], as_index=False).mean(numeric_only=True)

    # Full System (shipped): the SAME runs as Table 1 / Phase 2 certified eval (issue C1)
    p2 = pd.read_csv(args.phase2_aggregate.format(region=args.region))
    shipped = p2[p2.policy == "Learned Hierarchical"].copy()
    shipped["policy"] = "Full System (shipped ckpt)"
    agg = pd.concat([agg, shipped[agg.columns]], ignore_index=True)
    agg.to_csv(out / f"ablation_aggregate_{args.region}.csv", index=False)

    # Stats: mean/CI per row + Welch vs Full System (shipped) on seed means (n = 5)
    summary: dict = {
        "region": args.region,
        "protocol": {
            "train_seed": args.train_seed,
            "base_config": BASE_CFG,
            "eval_seeds": eval_seeds,
            "episodes_per_seed": args.episodes,
            "unit": "seed means",
            "full_system_source": "Phase 2 certified eval of results/runs/"
            f"checkpoint_hierarchical_{args.region}.pt (same runs as Table 1)",
        },
        "metrics": {},
        "comparisons_vs_full_shipped": {},
    }
    full = agg.loc[agg.policy == "Full System (shipped ckpt)"]
    for metric in ["WEL", "ISR", "CE"]:
        summary["metrics"][metric] = {}
        for policy in agg.policy.unique():
            v = agg.loc[agg.policy == policy, metric].to_numpy()
            lo, hi = bootstrap_ci(v)
            summary["metrics"][metric][policy] = {
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)),
                "ci_lo": lo,
                "ci_hi": hi,
                "per_seed": [round(float(x), 4) for x in v],
            }
    for metric in ["WEL", "ISR"]:
        summary["comparisons_vs_full_shipped"][metric] = {}
        f = full[metric].to_numpy()
        for policy in agg.policy.unique():
            if policy == "Full System (shipped ckpt)":
                continue
            v = agg.loc[agg.policy == policy, metric].to_numpy()
            summary["comparisons_vs_full_shipped"][metric][policy] = welch_and_d(v, f)

    with open(out / f"ablation_summary_{args.region}.json", "w") as fjson:
        json.dump(summary, fjson, indent=2)

    env.close()
    print(f"\nAblation study complete -> {out}")


if __name__ == "__main__":
    main()
