"""Evaluation script comparing learned hierarchical controller against heuristics and flat MARL.

ARCHITECTURAL NOTE (consistent with hierarchical_train.py):
- All STRATEGIC methods (No-Op, Greedy-Risk, Value-First, Learned Hierarchical) use
  TARGET-SEEKING low-level movement. This is fair because the frozen MAPPO actor ignores
  commander targets, making comparison biased if only the heuristics use target-seeking.
- Only "Flat MARL" uses the MAPPO actor with no strategic guidance.
- The commander's quality is measured purely by its sector selection decisions.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.agents.agent_networks import MAPPOActor
from wildfire_marl.agents.strategic_controller import StrategicController, get_sector_center
from wildfire_marl.train.hierarchical_train import extract_high_level_state
from wildfire_marl.eval.metrics import (
    burned_cells,
    infrastructure_survival_rate,
    weighted_economic_loss,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --- Low-level action helpers ------------------------------------------------

def target_seeking_action(env: MultiAgentFireEnv, agent: str, mask: list) -> int:
    """Move toward the target sector; suppress fire when at target."""
    y, x = env.agent_positions[agent]
    ty, tx = env.strategic_targets[agent]
    dy = ty - y
    dx = tx - x

    if abs(dy) >= abs(dx):
        if dy < 0 and len(mask) > 1 and mask[1]:
            return 1  # up
        elif dy > 0 and len(mask) > 2 and mask[2]:
            return 2  # down

    if dx < 0 and len(mask) > 3 and mask[3]:
        return 3  # left
    elif dx > 0 and len(mask) > 4 and mask[4]:
        return 4  # right

    if len(mask) > 5 and mask[5]:
        return 5  # suppress
    return 0  # noop


# --- Strategic Heuristics (sector dispatch) ----------------------------------

def get_greedy_risk_sectors(env: MultiAgentFireEnv) -> dict[str, tuple[int, int]]:
    """Greedy-Risk: dispatch agents to sectors with most active fire."""
    s_fire, _, _ = extract_high_level_state(env)
    s_fire_arr = s_fire.squeeze(0).numpy()
    sorted_sectors = np.argsort(-s_fire_arr)

    targets = {}
    for idx, agent in enumerate(env.agents):
        sec = sorted_sectors[idx % 16]
        targets[agent] = get_sector_center(sec)
    return targets


def get_value_first_sectors(env: MultiAgentFireEnv) -> dict[str, tuple[int, int]]:
    """Value-First: dispatch agents to sectors with high-value threatened assets."""
    s_fire, s_asset, _ = extract_high_level_state(env)
    s_fire_arr = s_fire.squeeze(0).numpy()
    s_asset_arr = s_asset.squeeze(0).numpy()

    risk_score = s_asset_arr * (s_fire_arr + 1.0)
    sorted_sectors = np.argsort(-risk_score)

    targets = {}
    for idx, agent in enumerate(env.agents):
        sec = sorted_sectors[idx % 16]
        targets[agent] = get_sector_center(sec)
    return targets


# --- Main evaluation function ------------------------------------------------

def evaluate_hierarchical_policy(
    env: MultiAgentFireEnv,
    setting: str,
    checkpoint_path: Path | None,
    num_episodes: int,
    base_seed: int,
    device: torch.device,
) -> dict[str, float]:

    # Load neural network models when available
    commander = None
    flat_marl_actor = None

    if setting == "Learned Hierarchical" and checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        commander = StrategicController(num_agents=env.num_agents).to(device)
        commander.load_state_dict(ckpt["commander_state_dict"])
        commander.eval()
    elif setting == "Flat MARL":
        region = "saudi" if "saudi" in str(env.env.map_dir).lower() else "california"
        flat_ckpt = Path(f"results/runs/checkpoint_mappo_{region}.pt")
        if flat_ckpt.exists():
            ckpt = torch.load(flat_ckpt, map_location=device)
            flat_marl_actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
            flat_marl_actor.load_state_dict(ckpt["actor_state_dict"])
            flat_marl_actor.eval()

    episode_rewards, episode_burned, episode_wel, episode_isr, episode_ce = [], [], [], [], []

    eval_seed_start = base_seed + 100000
    high_level_interval = 10

    # No compliance reward for clean eval
    env.apply_target_compliance = False

    for ep in range(num_episodes):
        seed = eval_seed_start + ep
        obs_dict, info_dict = env.reset(seed=seed)

        ep_rew = 0.0
        done = False
        step_count = 0
        curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}

        while not done:
            # 1. High-level Dispatch (every high_level_interval steps)
            if step_count % high_level_interval == 0:
                if setting == "No-Op":
                    # Stay in place
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
                        actions = [torch.argmax(l).item() for l in logits_list]
                    for idx, agent in enumerate(env.agents):
                        curr_targets[agent] = get_sector_center(actions[idx])
                elif setting == "Flat MARL":
                    # No strategic dispatch for flat MARL
                    curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}

            env.strategic_targets = curr_targets

            # 2. Low-level execution
            actions_dict = {}
            for agent in env.agents:
                mask = info_dict[agent]["action_mask"]
                if setting == "Flat MARL" and flat_marl_actor is not None:
                    # Flat MARL uses MAPPO (no target guidance)
                    with torch.no_grad():
                        obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32, device=device).unsqueeze(0)
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
                    # ALL strategic methods use target-seeking (fair comparison)
                    actions_dict[agent] = target_seeking_action(env, agent, mask)

            obs_dict, rewards_dict, terminations_dict, truncations_dict, info_dict = env.step(actions_dict)
            ep_rew += rewards_dict["agent_0"]
            done = terminations_dict["agent_0"] or truncations_dict["agent_0"]
            step_count += 1

        episode_rewards.append(ep_rew)
        final_state = env.env.fire_state
        episode_burned.append(burned_cells(final_state))

        wel = weighted_economic_loss(
            asset_type=env.env.asset_type,
            final_state=final_state,
            asset_values=env.env.asset_values,
        )
        isr = infrastructure_survival_rate(
            asset_type=env.env.asset_type,
            final_state=final_state,
        )
        episode_wel.append(wel)
        episode_isr.append(isr)
        episode_ce.append(info_dict[env.agents[0]].get("coordination_efficiency", 1.0))

    return {
        "reward_mean": float(np.mean(episode_rewards)),
        "burned_mean": float(np.mean(episode_burned)),
        "wel_mean": float(np.mean(episode_wel)),
        "isr_mean": float(np.mean(episode_isr)),
        "ce_mean": float(np.mean(episode_ce)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical dispatch policies.")
    parser.add_argument("--region", type=str, required=True, choices=["saudi", "california"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hierarchical_ckpt = Path(f"results/runs/checkpoint_hierarchical_{args.region}.pt")

    results = []

    configs = [
        ("No-Op", None),
        ("Greedy-Risk Heuristic", None),
        ("Value-First Heuristic", None),
        ("Flat MARL", None),
    ]
    if hierarchical_ckpt.exists():
        configs.append(("Learned Hierarchical", hierarchical_ckpt))

    for setting, ckpt_path in configs:
        print(f"Evaluating {setting}...")
        res = evaluate_hierarchical_policy(env, setting, ckpt_path, args.episodes, args.seed, device)
        res["policy"] = setting
        results.append(res)

    df = pd.DataFrame(results)
    df["region"] = args.region
    csv_path = Path(f"results/runs/hierarchical_eval_summary_{args.region}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"\nHierarchical evaluation summary saved to {csv_path}")
    print("\n" + "=" * 65)
    print(f"HIERARCHICAL EVALUATION REPORT: {args.region.upper()}")
    print("=" * 65)
    for _, row in df.iterrows():
        print(
            f"  {row['policy']:<25}: Reward {row['reward_mean']:8.2f} | "
            f"Burned {row['burned_mean']:6.1f} | "
            f"WEL {row['wel_mean']:6.1f} | "
            f"ISR {row['isr_mean']:.2f} | "
            f"CE {row['ce_mean']:.2f}"
        )
    print("=" * 65)

    env.close()


if __name__ == "__main__":
    main()
