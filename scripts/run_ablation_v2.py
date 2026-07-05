"""Ablation study evaluation script (v2).
Trains and evaluates ablated versions of the Hierarchical MARL system.
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
from scripts.run_transfer_v2 import evaluate_transfer_cell_v2

from wildfire_marl.agents.agent_networks import MAPPOActor
from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.train.hierarchical_train import train_hierarchical


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Run ablation study v2.")
    parser.add_argument("--region", type=str, required=True, choices=["saudi", "california"])
    parser.add_argument("--episodes", type=int, default=15, help="Eval episodes")
    parser.add_argument("--train_episodes", type=int, default=120, help="RL fine-tuning episodes")
    parser.add_argument("--pretrain_episodes", type=int, default=40, help="BC pretraining episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/phase8")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    map_name = "Saudi" if args.region.lower() == "saudi" else "California"
    data_dir = "data/cell2fire"
    infra_dir = f"{data_dir}/{map_name}"

    # Base config
    base_config = {
        "region": args.region,
        "seed": args.seed,
        "data_dir": data_dir,
        "pretrain_episodes": args.pretrain_episodes,
        "bc_epochs": 10 if args.pretrain_episodes <= 2 else 100,  # scale epochs down for smoke test
        "hierarchical_episodes": args.train_episodes,
        "kl_coef": 0.3,
        "ent_coef": 0.05,
        "isr_weight": 20.0,
        "gamma": 0.99,
        "rl_lr": 5e-5,
    }

    # Define ablated configurations
    ablations = {
        "Full Redesign": {},
        "Without BC Pretraining": {"pretrain_episodes": 0},
        "Without KL Regularization": {"kl_coef": 0.0},
        "Without Entropy Bonus": {"ent_coef": 0.0},
        "Without WEL/ISR Reward": {"use_raw_reward": True},
        "Without Target-Seeking": {"use_mappo_tactical": True},
    }

    results = []

    # Load low-level policy checkpoint
    low_level_ckpt = Path(f"results/runs/checkpoint_mappo_{args.region}.pt")
    if not low_level_ckpt.exists():
        raise FileNotFoundError(f"Missing MAPPO checkpoint: {low_level_ckpt}")

    for name, overrides in ablations.items():
        print("\n=========================================")
        print(f"Running Ablation: {name}")
        print("=========================================")

        cfg = base_config.copy()
        cfg.update(overrides)

        # Setup env
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

        # Load actor
        low_level_actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
        ckpt = torch.load(low_level_ckpt, map_location=device)
        low_level_actor.load_state_dict(ckpt["actor_state_dict"])

        # Train ablated commander
        commander = train_hierarchical(env, low_level_actor, cfg, device)

        # Save temp checkpoint for eval
        temp_ckpt_path = Path(args.output_dir) / f"temp_ablation_{args.region}.pt"
        torch.save(
            {
                "commander_state_dict": commander.state_dict(),
                "actor_state_dict": low_level_actor.state_dict(),
                "config": cfg,
            },
            temp_ckpt_path,
        )

        # Evaluate ablated policy
        eval_metrics = evaluate_transfer_cell_v2(
            env, "Learned Hierarchical", temp_ckpt_path, args.episodes, args.seed, device
        )
        env.close()

        results.append(
            {
                "Ablation Group": "Design Component",
                "Configuration": name,
                "Return": eval_metrics["reward_mean"],
                "Burned": eval_metrics["burned_mean"],
                "WEL": eval_metrics["wel_mean"],
                "ISR": eval_metrics["isr_mean"],
                "CE": eval_metrics["ce_mean"],
            }
        )

        # Remove temp checkpoint
        if temp_ckpt_path.exists():
            temp_ckpt_path.unlink()

    # Save ablation summary CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = out_dir / f"ablation_results_{args.region}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved ablation summary to {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
