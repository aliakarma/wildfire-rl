"""Cross-region evaluation and transfer robustness evaluation script."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wildfire_marl.agents.agent_networks import MAPPOActor
from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.eval.metrics import (
    burned_cells,
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.eval.transfer import adaptation_asymmetry, transfer_robustness_score
from wildfire_marl.viz.transfer import plot_transfer_heatmap


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_transfer_cell(
    env: MultiAgentFireEnv,
    actor_path: Path,
    num_episodes: int,
    base_seed: int,
    device: torch.device,
) -> dict[str, float]:
    actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
    ckpt = torch.load(actor_path, map_location=device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    episode_rewards = []
    episode_burned = []
    episode_wel = []
    episode_isr = []
    episode_ce = []

    eval_seed_start = base_seed + 100000

    for ep in range(num_episodes):
        seed = eval_seed_start + ep
        obs_dict, info_dict = env.reset(seed=seed)
        done = False

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

            obs_dict, rewards_dict, terminations_dict, truncations_dict, info_dict = env.step(
                actions_dict
            )
            ep_rew += rewards_dict["agent_0"]
            done = terminations_dict["agent_0"] or truncations_dict["agent_0"]

        # Collect metrics
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
    parser = argparse.ArgumentParser(description="Run cross-region transfer matrix evaluation.")
    parser.add_argument("--episodes", type=int, default=15, help="Number of transfer episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random base seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_saudi = Path("results/runs/checkpoint_mappo_saudi.pt")
    ckpt_cali = Path("results/runs/checkpoint_mappo_california.pt")

    if not ckpt_saudi.exists() or not ckpt_cali.exists():
        raise FileNotFoundError("Missing MAPPO checkpoints. Train MAPPO on both regions first!")

    # Environments
    env_saudi = MultiAgentFireEnv(
        num_agents=3,
        crop_size=9,
        coordination_penalty=0.1,
        fire_map="Saudi",
        data_dir="data/cell2fire",
        max_steps=150,
        steps_per_action=60,
        observe_infra=True,
        catastrophe_weight=2.0,
        cascade_prob=0.1,
        infra_dir="data/cell2fire/Saudi",
    )

    env_cali = MultiAgentFireEnv(
        num_agents=3,
        crop_size=9,
        coordination_penalty=0.1,
        fire_map="California",
        data_dir="data/cell2fire",
        max_steps=150,
        steps_per_action=60,
        observe_infra=True,
        catastrophe_weight=2.0,
        cascade_prob=0.1,
        infra_dir="data/cell2fire/California",
    )

    print("Evaluating symmetric transfer matrix cell-by-cell...")

    # 1. Saudi policy on Saudi (S -> S)
    print("Evaluating Saudi Policy on Saudi Environment...")
    res_s_s = evaluate_transfer_cell(env_saudi, ckpt_saudi, args.episodes, args.seed, device)

    # 2. Saudi policy on California (S -> C)
    print("Evaluating Saudi Policy on California Environment...")
    res_s_c = evaluate_transfer_cell(env_cali, ckpt_saudi, args.episodes, args.seed, device)

    # 3. California policy on California (C -> C)
    print("Evaluating California Policy on California Environment...")
    res_c_c = evaluate_transfer_cell(env_cali, ckpt_cali, args.episodes, args.seed, device)

    # 4. California policy on Saudi (C -> S)
    print("Evaluating California Policy on Saudi Environment...")
    res_c_s = evaluate_transfer_cell(env_saudi, ckpt_cali, args.episodes, args.seed, device)

    env_saudi.close()
    env_cali.close()

    # Calculate Transfer Robustness Scores (TRS)
    trs_s_c = transfer_robustness_score(res_c_c["reward_mean"], res_s_c["reward_mean"])
    trs_c_s = transfer_robustness_score(res_s_s["reward_mean"], res_c_s["reward_mean"])

    asymmetry = adaptation_asymmetry(trs_s_c, trs_c_s)

    # 2x2 Transfer Matrix of normalized scores
    matrix = np.array(
        [
            [1.0, trs_s_c],
            [trs_c_s, 1.0],
        ]
    )

    print("\n" + "=" * 60)
    print("CROSS-REGION TRANSFER RESULTS (REWARDS)")
    print("=" * 60)
    print(f"  Saudi -> Saudi     (S->S): {res_s_s['reward_mean']:8.2f} (Base)")
    print(f"  Saudi -> California (S->C): {res_s_c['reward_mean']:8.2f} (TRS: {trs_s_c:.2f})")
    print(f"  California -> Calif (C->C): {res_c_c['reward_mean']:8.2f} (Base)")
    print(f"  California -> Saudi (C->S): {res_c_s['reward_mean']:8.2f} (TRS: {trs_c_s:.2f})")
    print(f"  Adaptation Asymmetry (S->C vs C->S): {asymmetry:+.2f}")
    print("=" * 60)

    # Save to CSV
    transfer_results = [
        {"Source": "Saudi", "Target": "Saudi", **res_s_s, "TRS": 1.0},
        {"Source": "Saudi", "Target": "California", **res_s_c, "TRS": trs_s_c},
        {"Source": "California", "Target": "California", **res_c_c, "TRS": 1.0},
        {"Source": "California", "Target": "Saudi", **res_c_s, "TRS": trs_c_s},
    ]
    df = pd.DataFrame(transfer_results)
    csv_path = Path("results/runs/transfer_results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")

    # Plot transfer heatmap
    heatmap_path = Path("results/runs/transfer_heatmap.png")
    plot_transfer_heatmap(
        matrix_2x2=matrix,
        row_labels=["Saudi Policy", "California Policy"],
        col_labels=["Saudi Env", "California Env"],
        title="Transfer Robustness Scores (TRS)",
        save_path=heatmap_path,
    )
    print(f"Saved Transfer Heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()
