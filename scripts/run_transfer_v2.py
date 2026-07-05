"""Domain adaptation and transfer robustness evaluation script (v2) for Hierarchical MARL.
Evaluates Saudi and California policies (Hierarchical, Value-First, Flat MARL) cross-regionally.
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
from scripts.run_multiseed_eval import run_episode_eval

from wildfire_marl.agents.agent_networks import MAPPOActor
from wildfire_marl.agents.strategic_controller import StrategicController
from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.eval.transfer import (
    adaptation_asymmetry,
    cross_domain_gap,
    transfer_robustness_score,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_transfer_cell_v2(
    env: MultiAgentFireEnv,
    setting: str,
    ckpt_path: Path | None,
    num_episodes: int,
    base_seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluates a strategic setting on the given env, returning metric averages."""
    commander = None
    flat_marl_actor = None

    if setting == "Learned Hierarchical" and ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        commander = StrategicController(num_agents=env.num_agents).to(device)
        commander.load_state_dict(ckpt["commander_state_dict"])
        commander.eval()
    elif setting == "Flat MARL" and ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        flat_marl_actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
        flat_marl_actor.load_state_dict(ckpt["actor_state_dict"])
        flat_marl_actor.eval()

    episode_rewards = []
    episode_burned = []
    episode_wel = []
    episode_isr = []
    episode_ce = []

    eval_seed_start = base_seed + 200000

    for ep in range(num_episodes):
        seed = eval_seed_start + ep
        metrics, _, _ = run_episode_eval(env, setting, commander, flat_marl_actor, seed, device)
        episode_rewards.append(metrics["total_reward"])
        episode_burned.append(metrics["burned_cells"])
        episode_wel.append(metrics["WEL"])
        episode_isr.append(metrics["ISR"])
        episode_ce.append(metrics["CE"])

    return {
        "reward_mean": float(np.mean(episode_rewards)),
        "burned_mean": float(np.mean(episode_burned)),
        "wel_mean": float(np.mean(episode_wel)),
        "isr_mean": float(np.mean(episode_isr)),
        "ce_mean": float(np.mean(episode_ce)),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-region transfer runner v2.")
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/phase8")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Saudi checkpoints
    h_ckpt_saudi = Path("results/runs/checkpoint_hierarchical_saudi.pt")
    f_ckpt_saudi = Path("results/runs/checkpoint_mappo_saudi.pt")

    # California checkpoints
    h_ckpt_cali = Path("results/runs/checkpoint_hierarchical_california.pt")
    f_ckpt_cali = Path("results/runs/checkpoint_mappo_california.pt")

    # Setup environments
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

    results = []
    methods = ["Learned Hierarchical", "Value-First Heuristic", "Flat MARL"]

    for method in methods:
        print(f"\n--- Transfer Evaluation for {method} ---")

        # Determine checkpoints based on method
        if method == "Learned Hierarchical":
            ckpt_saudi = h_ckpt_saudi
            ckpt_cali = h_ckpt_cali
        elif method == "Flat MARL":
            ckpt_saudi = f_ckpt_saudi
            ckpt_cali = f_ckpt_cali
        else:
            ckpt_saudi = None
            ckpt_cali = None

        # 1. Saudi on Saudi (S -> S, Native)
        print("Evaluating Saudi Policy on Saudi Environment...")
        res_s_s = evaluate_transfer_cell_v2(
            env_saudi, method, ckpt_saudi, args.episodes, args.seed, device
        )

        # 2. Saudi on California (S -> C, Transfer)
        print("Evaluating Saudi Policy on California Environment...")
        res_s_c = evaluate_transfer_cell_v2(
            env_cali, method, ckpt_saudi, args.episodes, args.seed, device
        )

        # 3. California on California (C -> C, Native)
        print("Evaluating California Policy on California Environment...")
        res_c_c = evaluate_transfer_cell_v2(
            env_cali, method, ckpt_cali, args.episodes, args.seed, device
        )

        # 4. California on Saudi (C -> S, Transfer)
        print("Evaluating California Policy on Saudi Environment...")
        res_c_s = evaluate_transfer_cell_v2(
            env_saudi, method, ckpt_cali, args.episodes, args.seed, device
        )

        # Compute Transfer Robustness Scores (TRS) using reward_mean
        trs_s_c = transfer_robustness_score(res_c_c["reward_mean"], res_s_c["reward_mean"])
        trs_c_s = transfer_robustness_score(res_s_s["reward_mean"], res_c_s["reward_mean"])

        # Robustness gap (degradation)
        gap_s_c = cross_domain_gap(res_c_c["reward_mean"], res_s_c["reward_mean"])
        gap_c_s = cross_domain_gap(res_s_s["reward_mean"], res_c_s["reward_mean"])

        asymmetry = adaptation_asymmetry(trs_s_c, trs_c_s)

        results.append(
            {
                "policy": method,
                "S_to_S_reward": res_s_s["reward_mean"],
                "S_to_C_reward": res_s_c["reward_mean"],
                "C_to_C_reward": res_c_c["reward_mean"],
                "C_to_S_reward": res_c_s["reward_mean"],
                "trs_S_to_C": trs_s_c,
                "trs_C_to_S": trs_c_s,
                "gap_S_to_C": gap_s_c,
                "gap_C_to_S": gap_c_s,
                "asymmetry": asymmetry,
            }
        )

    env_saudi.close()
    env_cali.close()

    # Save transfer summary CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = out_dir / "transfer_results_v2.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved cross-region transfer results to {csv_path}")
    print(df.to_string(index=False))

    # Print LaTeX-ready tables
    print("\nLaTeX Transfer Robustness Table:")
    for _, row in df.iterrows():
        print(
            f"{row['policy']} & {row['S_to_S_reward']:.1f} & {row['S_to_C_reward']:.1f} ({row['trs_S_to_C']:.2f}) & "
            f"{row['C_to_C_reward']:.1f} & {row['C_to_S_reward']:.1f} ({row['trs_C_to_S']:.2f}) & "
            f"{row['asymmetry']:.2f} \\\\"
        )


if __name__ == "__main__":
    main()
