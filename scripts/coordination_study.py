"""Coordination study script comparing Shared vs Selfish rewards and plotting trajectories."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.eval.coordination import redundant_treatment_rate, spatial_division_of_labor
from wildfire_marl.eval.metrics import burned_cells
from wildfire_marl.train.marl_train import train_mappo
from wildfire_marl.viz.coordination import plot_agent_trajectories


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_eval_with_trajectory(
    env: MultiAgentFireEnv,
    actor_path: Path,
    device: torch.device,
    num_episodes: int = 10,
    seed_offset: int = 150000,
) -> dict[str, float]:
    from wildfire_marl.agents.agent_networks import MAPPOActor

    actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
    ckpt = torch.load(actor_path, map_location=device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    ep_rewards = []
    ep_burned = []
    ep_ce = []
    ep_dol = []
    ep_rtr = []

    # For visualization, we will capture trajectories of the first episode
    viz_agent_paths = None
    viz_fire_state = None

    for ep in range(num_episodes):
        seed = seed_offset + ep
        obs_dict, info_dict = env.reset(seed=seed)
        done = False

        agent_paths = {agent: [env.agent_positions[agent]] for agent in env.agents}
        total_treatments = 0
        redundant_treatments = 0
        ep_rew = 0.0

        while not done:
            actions_dict = {}
            for agent in env.agents:
                mask = info_dict[agent]["action_mask"]
                with torch.no_grad():
                    obs_t = torch.tensor(
                        obs_dict[agent], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                    logits = actor(obs_t, mask_t).squeeze(0)
                    prob = torch.softmax(logits, dim=-1)
                    prob[~torch.tensor(mask, device=device)] = 0.0
                    if prob.sum() > 0:
                        actions_dict[agent] = int(torch.argmax(prob).item())
                    else:
                        actions_dict[agent] = 0

            # Compute treatment metrics at current step
            unique_treatments = set()
            for agent, action in actions_dict.items():
                if action == 5:  # Treat
                    total_treatments += 1
                    pos = env.agent_positions[agent]
                    if pos in unique_treatments:
                        redundant_treatments += 1
                    else:
                        unique_treatments.add(pos)

            obs_dict, rewards_dict, terminations_dict, truncations_dict, info_dict = env.step(
                actions_dict
            )
            ep_rew += rewards_dict["agent_0"]
            done = terminations_dict["agent_0"] or truncations_dict["agent_0"]

            for agent in env.agents:
                agent_paths[agent].append(env.agent_positions[agent])

        ep_rewards.append(ep_rew)
        ep_burned.append(burned_cells(env.env.fire_state))
        ep_ce.append(info_dict[env.agents[0]].get("coordination_efficiency", 1.0))
        ep_dol.append(spatial_division_of_labor(agent_paths))
        ep_rtr.append(redundant_treatment_rate(total_treatments, redundant_treatments))

        if ep == 0:
            viz_agent_paths = agent_paths
            viz_fire_state = env.env.fire_state

    return (
        {
            "reward_mean": float(np.mean(ep_rewards)),
            "burned_mean": float(np.mean(ep_burned)),
            "ce_mean": float(np.mean(ep_ce)),
            "dol_mean": float(np.mean(ep_dol)),
            "rtr_mean": float(np.mean(ep_rtr)),
        },
        viz_agent_paths,
        viz_fire_state,
    )


def main():
    parser = argparse.ArgumentParser(description="Run Shared vs Selfish Reward ablation study.")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps for selfish agent")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Coordination Ablation Study on {device}...")

    # Configurations
    shared_ckpt = Path("results/runs/checkpoint_mappo_saudi.pt")
    selfish_ckpt = Path("results/runs/checkpoint_mappo_saudi_selfish.pt")

    # Saudi Region Setup
    data_dir = "data/cell2fire"
    map_name = "Saudi"
    infra_dir = f"{data_dir}/{map_name}"

    # 1. Train Selfish Agent if checkpoint doesn't exist
    if not selfish_ckpt.exists():
        print("\n--- Training Selfish MAPPO (coordination_penalty = 0.0) ---")
        env_selfish = MultiAgentFireEnv(
            num_agents=3,
            crop_size=9,
            coordination_penalty=0.0,  # Selfish reward
            fire_map=map_name,
            data_dir=data_dir,
            max_steps=150,
            steps_per_action=60,
            observe_infra=True,
            catastrophe_weight=2.0,
            cascade_prob=0.1,
            infra_dir=infra_dir,
        )

        cfg_selfish = {
            "region": "saudi",
            "algo": "mappo",
            "total_steps": args.steps,
            "n_steps": 1024,
            "batch_size": 64,
            "ppo_epochs": 4,
            "clip_eps": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "seed": 42,
        }

        actor_s, critic_s = train_mappo(env_selfish, cfg_selfish, device)
        torch.save(
            {
                "actor_state_dict": actor_s.state_dict(),
                "critic_state_dict": critic_s.state_dict(),
                "config": cfg_selfish,
            },
            selfish_ckpt,
        )
        env_selfish.close()
        print("Selfish MAPPO policy saved successfully.")

    # 2. Evaluate Shared vs Selfish Policies
    print("\n--- Evaluating Policies ---")

    # Shared Env
    env_eval = MultiAgentFireEnv(
        num_agents=3,
        crop_size=9,
        coordination_penalty=0.1,  # Evaluation penalty standard
        fire_map=map_name,
        data_dir=data_dir,
        max_steps=150,
        steps_per_action=60,
        observe_infra=True,
        catastrophe_weight=2.0,
        cascade_prob=0.1,
        infra_dir=infra_dir,
    )

    # Eval Shared
    shared_res, shared_paths, shared_fire = run_eval_with_trajectory(env_eval, shared_ckpt, device)

    # Eval Selfish
    selfish_res, selfish_paths, selfish_fire = run_eval_with_trajectory(
        env_eval, selfish_ckpt, device
    )

    env_eval.close()

    # 3. Plot trajectories
    print("\n--- Generating Trajectory Plots ---")
    shared_plot_path = Path("results/runs/trajectories_shared_saudi.png")
    selfish_plot_path = Path("results/runs/trajectories_selfish_saudi.png")

    plot_agent_trajectories(32, 32, shared_paths, shared_fire, shared_plot_path)
    plot_agent_trajectories(32, 32, selfish_paths, selfish_fire, selfish_plot_path)
    print(f"Plots saved to:\n - {shared_plot_path}\n - {selfish_plot_path}")

    # 4. Summarize and Print Results
    results = [
        {"Setting": "Shared Reward (Cooperative)", **shared_res},
        {"Setting": "Selfish Reward (No Penalty)", **selfish_res},
    ]

    df = pd.DataFrame(results)
    csv_path = Path("results/runs/coordination_study_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

    print("\n" + "=" * 60)
    print("COORDINATION ABLATION STUDY RESULTS")
    print("=" * 60)
    for _, row in df.iterrows():
        print(
            f"  {row['Setting']:<30}: Return {row['reward_mean']:8.2f} | "
            f"Burned {row['burned_mean']:6.1f} | "
            f"CE {row['ce_mean']:.2f} | "
            f"DoL {row['dol_mean']:.2f} | "
            f"RTR {row['rtr_mean']:.2%}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
