"""Multi-seed evaluation runner script.
Evaluates hierarchical policies and baselines over multiple independent seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import random

import numpy as np
import pandas as pd
import torch
from scripts.eval_hierarchical import (
    get_greedy_risk_sectors,
    get_value_first_sectors,
    target_seeking_action,
)

from wildfire_marl.agents.agent_networks import MAPPOActor
from wildfire_marl.agents.strategic_controller import StrategicController, get_sector_center
from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.eval.metrics import (
    burned_cells,
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.train.hierarchical_train import extract_high_level_state


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_episode_eval(
    env: MultiAgentFireEnv,
    setting: str,
    commander: StrategicController | None,
    flat_marl_actor: MAPPOActor | None,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]]]:
    """Runs a single episode evaluation, tracking agent positions and targets history."""
    obs_dict, info_dict = env.reset(seed=seed)

    positions_history = {agent: [env.agent_positions[agent]] for agent in env.agents}
    targets_history = {agent: [env.agent_positions[agent]] for agent in env.agents}

    done = False
    step_count = 0
    high_level_interval = 10
    ep_reward = 0.0
    curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}
    env.apply_target_compliance = False

    while not done:
        # High-level dispatch
        if step_count % high_level_interval == 0:
            if setting == "No-Op":
                curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}
            elif setting == "Greedy-Risk Heuristic":
                curr_targets = get_greedy_risk_sectors(env)
            elif setting == "Value-First Heuristic":
                curr_targets = get_value_first_sectors(env)
            elif setting == "Learned Hierarchical" and commander is not None:
                s_fire, s_asset, s_agent = extract_high_level_state(env)
                s_fire = s_fire.to(device)
                s_asset = s_asset.to(device)
                s_agent = s_agent.to(device)
                with torch.no_grad():
                    logits_list = commander(s_fire, s_asset, s_agent)
                    actions = [torch.argmax(logits).item() for logits in logits_list]
                for idx, agent in enumerate(env.agents):
                    curr_targets[agent] = get_sector_center(actions[idx])
            elif setting == "Flat MARL":
                curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}

        env.strategic_targets = curr_targets

        # Low-level execution
        actions_dict = {}
        for agent in env.agents:
            mask = info_dict[agent]["action_mask"]
            if setting == "Flat MARL" and flat_marl_actor is not None:
                with torch.no_grad():
                    obs_t = torch.tensor(
                        obs_dict[agent], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                    logits = flat_marl_actor(obs_t, mask_t).squeeze(0)
                    prob = torch.softmax(logits, dim=-1)
                    prob = prob * torch.tensor(mask, device=device).float()
                    if prob.sum() > 0:
                        actions_dict[agent] = int(torch.argmax(prob).item())
                    else:
                        actions_dict[agent] = 0
            elif setting == "No-Op":
                actions_dict[agent] = 0
            else:
                actions_dict[agent] = target_seeking_action(env, agent, mask)

        obs_dict, rewards_dict, terminations_dict, truncations_dict, info_dict = env.step(
            actions_dict
        )
        ep_reward += rewards_dict["agent_0"]
        done = terminations_dict["agent_0"] or truncations_dict["agent_0"]

        # Record histories
        for agent in env.agents:
            positions_history[agent].append(env.agent_positions[agent])
            targets_history[agent].append(env.strategic_targets[agent])

        step_count += 1

    final_state = env.env.fire_state
    wel = weighted_economic_loss(
        asset_type=env.env.asset_type,
        final_state=final_state,
        asset_values=env.env.asset_values,
    )
    isr = infrastructure_survival_rate(
        asset_type=env.env.asset_type,
        final_state=final_state,
    )
    burned = burned_cells(final_state)
    ce = info_dict[env.agents[0]].get("coordination_efficiency", 1.0)

    metrics = {
        "WEL": wel,
        "ISR": isr,
        "CE": ce,
        "burned_cells": burned,
        "total_reward": ep_reward,
    }
    return metrics, positions_history, targets_history


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed evaluation.")
    parser.add_argument("--region", type=str, required=True, choices=["saudi", "california"])
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/phase8")
    args = parser.parse_args()

    set_seed(args.base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    map_name = "Saudi" if args.region.lower() == "saudi" else "California"
    data_dir = "data/cell2fire"
    infra_dir = f"{data_dir}/{map_name}"

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
        infra_dir=infra_dir,
    )

    hierarchical_ckpt = Path(f"results/runs/checkpoint_hierarchical_{args.region}.pt")
    flat_ckpt = Path(f"results/runs/checkpoint_mappo_{args.region}.pt")

    commander = None
    if hierarchical_ckpt.exists():
        ckpt = torch.load(hierarchical_ckpt, map_location=device)
        commander = StrategicController(num_agents=env.num_agents).to(device)
        commander.load_state_dict(ckpt["commander_state_dict"])
        commander.eval()
        print(f"Loaded hierarchical controller from {hierarchical_ckpt}")

    flat_marl_actor = None
    if flat_ckpt.exists():
        ckpt = torch.load(flat_ckpt, map_location=device)
        flat_marl_actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
        flat_marl_actor.load_state_dict(ckpt["actor_state_dict"])
        flat_marl_actor.eval()
        print(f"Loaded flat MARL actor from {flat_ckpt}")

    policies = ["No-Op", "Greedy-Risk Heuristic", "Value-First Heuristic", "Flat MARL"]
    if commander is not None:
        policies.append("Learned Hierarchical")

    records = []

    # Generate list of seeds deterministically
    eval_seeds = [args.base_seed + s * 1000 for s in range(args.seeds)]

    for seed in eval_seeds:
        for policy in policies:
            print(f"Evaluating {policy} with Seed {seed} for {args.episodes} episodes...")
            # We run multiple episodes per seed, and average them, or record each episode as a separate row.
            # To be absolutely rigorous, let's record each episode result as a row!
            for ep in range(args.episodes):
                ep_seed = seed + ep
                metrics, pos_hist, tar_hist = run_episode_eval(
                    env, policy, commander, flat_marl_actor, ep_seed, device
                )
                records.append(
                    {
                        "seed": seed,
                        "episode": ep,
                        "region": args.region,
                        "policy": policy,
                        **metrics,
                    }
                )

    env.close()

    # Save per-run (episode-level) CSV
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    per_run_csv = out_path / f"multiseed_eval_raw_{args.region}.csv"
    df.to_csv(per_run_csv, index=False)
    print(f"Saved raw episode results to {per_run_csv}")

    # Compute aggregate stats by seed & policy, then across seeds
    df_agg = df.groupby(["region", "policy", "seed"]).mean(numeric_only=True).reset_index()
    agg_csv = out_path / f"multiseed_eval_aggregate_{args.region}.csv"
    df_agg.to_csv(agg_csv, index=False)
    print(f"Saved aggregated seed-level results to {agg_csv}")


if __name__ == "__main__":
    main()
