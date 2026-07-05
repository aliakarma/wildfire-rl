"""Evaluation runner for multi-agent policies (MAPPO, QMIX) against heuristics.

Computes average rewards, burned cells, economic loss, and coordination efficiency.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.env.rewards import InfrastructureWeightedReward
from wildfire_marl.agents.agent_networks import MAPPOActor, QMIXAgent
from wildfire_marl.eval.metrics import (
    burned_cells,
    infrastructure_survival_rate,
    weighted_economic_loss,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_policy(
    env: MultiAgentFireEnv,
    policy_type: str,
    checkpoint_path: Path | None,
    num_episodes: int,
    base_seed: int,
    device: torch.device,
) -> dict[str, float]:
    num_agents = env.num_agents
    crop_size = env.crop_size

    # Load neural network models if checkpoint is provided
    actor = None
    q_agent = None

    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "actor_state_dict" in ckpt:
            actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
            actor.load_state_dict(ckpt["actor_state_dict"])
            actor.eval()
            print(f"Loaded MAPPO Actor checkpoint from {checkpoint_path}")
        elif "agent_state_dict" in ckpt:
            q_agent = QMIXAgent(in_channels=8, action_dim=6, features_dim=64).to(device)
            q_agent.load_state_dict(ckpt["agent_state_dict"])
            q_agent.eval()
            print(f"Loaded QMIX Agent checkpoint from {checkpoint_path}")

    episode_rewards = []
    episode_burned = []
    episode_wel = []
    episode_isr = []
    episode_ce = []

    # Shift evaluation seeds by 100,000 to ensure zero overlap with training
    eval_seed_start = base_seed + 100000

    for ep in range(num_episodes):
        seed = eval_seed_start + ep
        obs_dict, info_dict = env.reset(seed=seed)

        ep_rew = 0.0
        done = False

        while not done:
            actions_dict = {}
            for agent in env.agents:
                mask = info_dict[agent]["action_mask"]

                if actor is not None:
                    # MAPPO greedy action
                    with torch.no_grad():
                        obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32, device=device).unsqueeze(0)
                        mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                        logits = actor(obs_t, mask_t).squeeze(0)
                        prob = torch.softmax(logits, dim=-1)
                        # Filter invalid
                        prob[~torch.tensor(mask, device=device)] = 0.0
                        if prob.sum() > 0:
                            actions_dict[agent] = int(torch.argmax(prob).item())
                        else:
                            actions_dict[agent] = 0
                elif q_agent is not None:
                    # QMIX greedy action
                    with torch.no_grad():
                        obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32, device=device).unsqueeze(0)
                        q_vals = q_agent(obs_t).squeeze(0).cpu().numpy()
                    q_vals[~mask] = -1e9
                    actions_dict[agent] = int(np.argmax(q_vals))
                else:
                    # Heuristic baseline: value-weighted frontier priority if possible,
                    # else random valid suppression action
                    y, x = env.agent_positions[agent]
                    # We can use simple rule: if can treat, treat!
                    if mask[5]:
                        actions_dict[agent] = 5 # Treat
                    else:
                        # Otherwise move random direction
                        valid_movements = [0, 1, 2, 3, 4]
                        actions_dict[agent] = random.choice(valid_movements)

            obs_dict, rewards_dict, terminations_dict, truncations_dict, info_dict = env.step(actions_dict)
            ep_rew += rewards_dict["agent_0"]
            done = terminations_dict["agent_0"] or truncations_dict["agent_0"]

        # Collect metrics
        episode_rewards.append(ep_rew)
        final_state = env.env.fire_state
        episode_burned.append(burned_cells(final_state))
        
        # Calculate strategic economic values
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
        "reward_std": float(np.std(episode_rewards)),
        "burned_mean": float(np.mean(episode_burned)),
        "burned_std": float(np.std(episode_burned)),
        "wel_mean": float(np.mean(episode_wel)),
        "wel_std": float(np.std(episode_wel)),
        "isr_mean": float(np.mean(episode_isr)),
        "ce_mean": float(np.mean(episode_ce)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained cooperating MARL policies.")
    parser.add_argument("--region", type=str, required=True, choices=["saudi", "california"], help="saudi or california")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random base seed")
    args = parser.parse_args()

    set_seed(args.seed)

    map_name = "Saudi" if args.region.lower() == "saudi" else "California"
    data_dir = "data/cell2fire"
    infra_dir = f"{data_dir}/{map_name}"

    # Initialize environment
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

    results = []

    # 1. Evaluate Heuristic Baseline
    print("Evaluating Multi-Agent Heuristic Policy...")
    h_res = evaluate_policy(env, "Heuristic", None, args.episodes, args.seed, device)
    h_res["policy"] = "Heuristic"
    results.append(h_res)

    # 2. Evaluate MAPPO Checkpoint
    mappo_path = Path(f"results/runs/checkpoint_mappo_{args.region}.pt")
    if mappo_path.exists():
        print("Evaluating MAPPO Policy...")
        mappo_res = evaluate_policy(env, "MAPPO", mappo_path, args.episodes, args.seed, device)
        mappo_res["policy"] = "MAPPO"
        results.append(mappo_res)

    # 3. Evaluate QMIX Checkpoint
    qmix_path = Path(f"results/runs/checkpoint_qmix_{args.region}.pt")
    if qmix_path.exists():
        print("Evaluating QMIX Policy...")
        qmix_res = evaluate_policy(env, "QMIX", qmix_path, args.episodes, args.seed, device)
        qmix_res["policy"] = "QMIX"
        results.append(qmix_res)

    # Save results summary
    df = pd.DataFrame(results)
    df["region"] = args.region
    csv_path = Path(f"results/runs/marl_eval_summary_{args.region}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"\nEvaluation summary saved to {csv_path}")
    print("\n" + "=" * 50)
    print(f"MARL EVALUATION REPORT: {args.region.upper()}")
    print("=" * 50)
    for _, row in df.iterrows():
        print(
            f"  {row['policy']:<12}: Reward {row['reward_mean']:8.2f} | "
            f"Burned {row['burned_mean']:6.1f} | "
            f"WEL {row['wel_mean']:6.1f} | "
            f"ISR {row['isr_mean']:.2f} | "
            f"CE {row['ce_mean']:.2f}"
        )

    env.close()


if __name__ == "__main__":
    main()
