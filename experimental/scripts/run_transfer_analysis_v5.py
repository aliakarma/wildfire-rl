#!/usr/bin/env python
"""Phase 5 cross-regional transfer evaluation driver.

Evaluates 5 and 10 agent Hybrid models on Saudi Arabia and California,
calculates transfer degradation, statistical significance, and generates
transfer learning comparison charts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from wildfire_rl.config import load_config
from wildfire_rl.envs.hybrid_multi_agent import HybridMultiAgentWildfireEnv
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.loading import load_ppo_model
from wildfire_rl.eval.significance import welch_ttest, confidence_interval_95, cohens_d
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir
from wildfire_rl.logging_utils import get_logger

logger = get_logger("run_transfer_analysis_v5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run in lightweight smoke test mode")
    args = ap.parse_args()

    n_episodes = 2 if args.smoke else 20
    seeds = [0] if args.smoke else [0, 1, 2]

    cfg = load_config("configs/ppo/hybrid_marl.yaml")
    agent_counts = [5, 10]

    v5_results_dir = ensure_dir(results_dir() / "v5")
    v5_figs_dir = ensure_dir(Path("figures") / "v5")

    saudi_tensor = np.load(region_tensor_path("saudi_eastern_province", 32))
    california_tensor = np.load(region_tensor_path("california", 32))

    transfer_results = []
    logger.info("Starting Phase 5 Cross-Regional Generalization Analysis...")

    pairs = [
        ("saudi", "saudi", saudi_tensor),
        ("saudi", "california", california_tensor),
        ("california", "california", california_tensor),
        ("california", "saudi", saudi_tensor),
    ]

    for train_reg, test_reg, tensor in pairs:
        for n_agents in agent_counts:
            logger.info(f"Evaluating transfer {train_reg} -> {test_reg} (N={n_agents})...")
            
            # Use Hybrid Environment with nearest_fire routing
            env_cfg = cfg.env
            env_cfg.reward_mode = "normalized"
            env_cfg.reward_v3.enabled = True

            factory = lambda t=tensor, c=env_cfg, na=n_agents: HybridMultiAgentWildfireEnv(
                state_tensor=t, config=c, num_agents=na, routing_strategy="nearest_fire"
            )

            rewards = []
            burned = []
            fires = []
            for seed in seeds:
                model_name = f"ppo_hybrid_marl_{train_reg}_{n_agents}agents_nearest_fire_seed_{seed}.zip"
                model_path = models_dir() / model_name
                if not model_path.exists():
                    logger.warning(f"Model checkpoint missing: {model_name}. Skipping.")
                    continue

                try:
                    model = load_ppo_model(model_path, factory())
                    res = evaluate_policy(
                        model, factory, n_episodes=n_episodes,
                        base_seed=0, deterministic=True, metrics_cfg=cfg.metrics,
                    )
                    rewards.extend([ep["episode_reward"] for ep in res["episodes"]])
                    burned.extend([ep["burned_cells"] for ep in res["episodes"]])
                    fires.extend([ep["fire_intensity"] for ep in res["episodes"]])
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name}: {e}")

            if rewards:
                reward_arr = np.array(rewards)
                burned_arr = np.array(burned)
                fire_arr = np.array(fires)
                ci_lo, ci_hi = confidence_interval_95(reward_arr)

                transfer_results.append({
                    "train_region": train_reg,
                    "test_region": test_reg,
                    "num_agents": n_agents,
                    "reward_mean": float(reward_arr.mean()),
                    "reward_std": float(reward_arr.std()),
                    "reward_ci_lo": float(ci_lo),
                    "reward_ci_hi": float(ci_hi),
                    "burned_cells_mean": float(burned_arr.mean()),
                    "fire_intensity_mean": float(fire_arr.mean()),
                })

    if not transfer_results:
        logger.warning("No transfer results evaluated. Exiting.")
        return

    df_trans = pd.DataFrame(transfer_results)
    
    # Calculate degradation percentage
    degradations = []
    for _, row in df_trans.iterrows():
        train = row["train_region"]
        test = row["test_region"]
        n = row["num_agents"]
        
        if train == test:
            degradations.append(0.0)
        else:
            # Native reward
            native_row = df_trans[(df_trans["train_region"] == test) & 
                                  (df_trans["test_region"] == test) & 
                                  (df_trans["num_agents"] == n)]
            if not native_row.empty:
                native_mean = native_row["reward_mean"].values[0]
                degrad = (native_mean - row["reward_mean"]) / abs(native_mean) * 100
                degradations.append(float(degrad))
            else:
                degradations.append(np.nan)

    df_trans["transfer_degradation_pct"] = degradations
    df_trans.to_csv(v5_results_dir / "marl_transfer_statistics_v5.csv", index=False)
    logger.info("Saved Phase 5 transfer stats -> results/v5/marl_transfer_statistics_v5.csv")

    # Generate Figures
    generate_transfer_plots(df_trans, v5_figs_dir)


def generate_transfer_plots(df: pd.DataFrame, figs_dir: Path):
    """Plot Phase 5 transfer matrices and degradation curves."""
    sns.set_theme(style="whitegrid")

    for n in df["num_agents"].unique():
        sub = df[df["num_agents"] == n]
        pivot = sub.pivot(index="train_region", columns="test_region", values="reward_mean")

        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f", cbar_kws={'label': 'Normalized Reward'})
        plt.title(f"Hybrid PPO Transfer Matrix (N={n} Agents)", fontsize=12, fontweight="bold")
        plt.savefig(figs_dir / f"hybrid_transfer_heatmap_N{n}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot degradation bar chart
    df_deg = df[df["train_region"] != df["test_region"]]
    if not df_deg.empty:
        plt.figure(figsize=(8, 5))
        df_deg["transfer_label"] = df_deg["train_region"].str.title() + " -> " + df_deg["test_region"].str.title()
        sns.barplot(data=df_deg, x="transfer_label", y="transfer_degradation_pct", hue="num_agents", palette="dark")
        plt.ylabel("Reward Degradation (%)", fontsize=11)
        plt.xlabel("Transfer Path", fontsize=11)
        plt.title("Cross-Regional Generalization Degradation Curves", fontsize=12, fontweight="bold")
        plt.savefig(figs_dir / "transfer_degradation_curves_v5.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated transfer plots in figures/v5/")


if __name__ == "__main__":
    main()
