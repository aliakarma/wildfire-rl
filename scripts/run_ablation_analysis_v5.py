#!/usr/bin/env python
"""Phase 5 ablation evaluation script.

Compares Hybrid PPO coordination against Pure NearestFire, Static Partitioning,
Random Territory Assignment, and Routing-Only baselines.
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
from wildfire_rl.envs.multi_agent_v3 import MultiAgentWildfireEnvV3
from wildfire_rl.eval.baselines import NearestFirePolicy
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.loading import load_ppo_model
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir
from wildfire_rl.logging_utils import get_logger

logger = get_logger("run_ablation_analysis_v5")


# Custom baseline policies for hybrid environment targets
class StaticPartitioningPolicy:
    """Always assigns each agent to a fixed quadrant target sector."""
    def __init__(self, num_agents: int):
        self.num_agents = num_agents

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        # Assign agent i to sector i % 4, with any overflow targeting global nearest (4)
        actions = [i % 4 if i < 4 else 4 for i in range(self.num_agents)]
        return np.array(actions), None


class RandomTerritoryPolicy:
    """Randomly assigns each agent a target quadrant sector at each step."""
    def __init__(self, num_agents: int):
        self.num_agents = num_agents

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        # Random choice between NW, NE, SW, SE, Global Nearest (0-4)
        actions = np.random.randint(0, 5, size=self.num_agents)
        return actions, None


class RoutingOnlyPolicy:
    """All agents always target Global Nearest active fire (equivalent to no coordination)."""
    def __init__(self, num_agents: int):
        self.num_agents = num_agents

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        # All target Global Nearest (4)
        actions = np.full(self.num_agents, 4)
        return actions, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run in lightweight smoke test mode")
    args = ap.parse_args()

    n_episodes = 2 if args.smoke else 20
    seeds = [0] if args.smoke else [0, 1, 2]

    cfg = load_config("configs/experiment/phase5_ablation.yaml")
    v5_results_dir = ensure_dir(results_dir() / "v5")
    v5_figs_dir = ensure_dir(Path("figures") / "v5")

    tensor = np.load(region_tensor_path("saudi_eastern_province", 32))
    n_agents = 5
    logger.info("Starting Phase 5 Ablation Analysis (Saudi, 5 Agents)...")

    # Define environment factories
    env_cfg = cfg.env
    env_cfg.reward_mode = "normalized"
    env_cfg.reward_v3.enabled = True

    factory_hybrid = lambda: HybridMultiAgentWildfireEnv(
        state_tensor=tensor, config=env_cfg, num_agents=n_agents, routing_strategy="nearest_fire"
    )
    factory_v3 = lambda: MultiAgentWildfireEnvV3(
        state_tensor=tensor, config=env_cfg, num_agents=n_agents
    )

    ablation_results = []

    # 1. Pure NearestFire (Standard Heuristic)
    logger.info("Evaluating Pure NearestFire...")
    near_policy = NearestFirePolicy(action_space=factory_v3().action_space, grid_size=32)
    res = evaluate_policy(near_policy, factory_v3, n_episodes=n_episodes, base_seed=0, deterministic=True, metrics_cfg=cfg.metrics)
    for ep in res["episodes"]:
        ablation_results.append({"baseline": "Pure NearestFire", "reward": ep["episode_reward"], "fire_intensity": ep["fire_intensity"]})

    # 2. Static Quadrant Partitioning
    logger.info("Evaluating Static Quadrant Partitioning...")
    static_policy = StaticPartitioningPolicy(n_agents)
    res = evaluate_policy(static_policy, factory_hybrid, n_episodes=n_episodes, base_seed=0, deterministic=True, metrics_cfg=cfg.metrics)
    for ep in res["episodes"]:
        ablation_results.append({"baseline": "Static Partitioning", "reward": ep["episode_reward"], "fire_intensity": ep["fire_intensity"]})

    # 3. Random Territory Assignment
    logger.info("Evaluating Random Territory Assignment...")
    random_policy = RandomTerritoryPolicy(n_agents)
    res = evaluate_policy(random_policy, factory_hybrid, n_episodes=n_episodes, base_seed=0, deterministic=True, metrics_cfg=cfg.metrics)
    for ep in res["episodes"]:
        ablation_results.append({"baseline": "Random Territory", "reward": ep["episode_reward"], "fire_intensity": ep["fire_intensity"]})

    # 4. Routing-Only
    logger.info("Evaluating Routing-Only (Global Target)...")
    routing_policy = RoutingOnlyPolicy(n_agents)
    res = evaluate_policy(routing_policy, factory_hybrid, n_episodes=n_episodes, base_seed=0, deterministic=True, metrics_cfg=cfg.metrics)
    for ep in res["episodes"]:
        ablation_results.append({"baseline": "Routing Only", "reward": ep["episode_reward"], "fire_intensity": ep["fire_intensity"]})

    # 5. Full Hybrid PPO
    logger.info("Evaluating Full Hybrid PPO...")
    for seed in seeds:
        model_name = f"ppo_hybrid_marl_saudi_5agents_nearest_fire_seed_{seed}.zip"
        model_path = models_dir() / model_name
        if model_path.exists():
            try:
                model = load_ppo_model(model_path, factory_hybrid())
                res = evaluate_policy(model, factory_hybrid, n_episodes=n_episodes, base_seed=0, deterministic=True, metrics_cfg=cfg.metrics)
                for ep in res["episodes"]:
                    ablation_results.append({"baseline": "Full Hybrid PPO", "reward": ep["episode_reward"], "fire_intensity": ep["fire_intensity"]})
            except Exception as e:
                logger.error(f"Failed to load Hybrid PPO in ablation: {e}")

    df_ab = pd.DataFrame(ablation_results)
    df_ab.to_csv(v5_results_dir / "routing_vs_coordination_ablation.csv", index=False)
    logger.info("Saved Phase 5 ablation table -> results/v5/routing_vs_coordination_ablation.csv")

    # Generate Figures
    generate_ablation_plots(df_ab, v5_figs_dir)


def generate_ablation_plots(df: pd.DataFrame, figs_dir: Path):
    """Plot Phase 5 ablation comparison bar charts."""
    sns.set_theme(style="whitegrid")

    # Figure 1: Routing vs RL Ablation
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="baseline", y="fire_intensity", palette="Set2", errorbar="ci")
    plt.ylabel("Final Fire Intensity (Lower is Better)", fontsize=11)
    plt.xlabel("Ablation Baseline Configuration", fontsize=11)
    plt.title("Routing vs RL Coordination Ablation Study (Saudi, 5 Agents)", fontsize=12, fontweight="bold")
    plt.savefig(figs_dir / "routing_vs_rl_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: Territorial Partitioning Ablation (Normalized Reward)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="baseline", y="reward", palette="coolwarm", errorbar="ci")
    plt.ylabel("Normalized Return Reward (Higher is Better)", fontsize=11)
    plt.xlabel("Ablation Baseline Configuration", fontsize=11)
    plt.title("Territorial Partitioning & Coordination Component Analysis", fontsize=12, fontweight="bold")
    plt.savefig(figs_dir / "territorial_partitioning_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated ablation plots in figures/v5/")


if __name__ == "__main__":
    main()
