#!/usr/bin/env python
"""Phase 3 evaluation driver: MARL Coordination Redesign & True Stabilization.

Evaluates PPO-V1, PPO-V2, and PPO-V3 models against baselines. Computes advanced
coordination metrics (overlap efficiency, suppression partitioning, frontier coverage,
specialization emergence), and generates comparative publication-quality figures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import convolve

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from wildfire_rl.config import load_config
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv
from wildfire_rl.envs.multi_agent_v2 import MultiAgentWildfireEnvV2
from wildfire_rl.envs.multi_agent_v3 import MultiAgentWildfireEnvV3
from wildfire_rl.eval.baselines import (
    NoOpPolicy, RandomPolicy, NearestFirePolicy, FrontierPolicy,
)
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.loading import load_ppo_model
from wildfire_rl.eval.significance import welch_ttest, confidence_interval_95, cohens_d
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir
from wildfire_rl.logging_utils import get_logger

logger = get_logger("run_marl_v3_evaluation")

_NEIGHBOR_KERNEL = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def run_detailed_diagnostics_v3(
    model: Any, env_factory: Any, seed: int = 0
) -> dict[str, Any]:
    """Run one episode to collect coordination and spatial diagnostics."""
    env = env_factory()
    if hasattr(model, "env"):
        model.env = env

    obs, _ = env.reset(seed=seed)
    done = False

    n_agents = env.num_agents
    visit_maps = [np.zeros((32, 32)) for _ in range(n_agents)]
    suppression_maps = [np.zeros((32, 32)) for _ in range(n_agents)]
    actions_taken = []
    fire_sizes = [float(env.state[0].sum())]
    agent_paths = [[] for _ in range(n_agents)]
    
    # Track frontier cells across the episode
    total_frontier_cells_count = 0
    suppressed_frontier_cells_count = 0

    for i in range(n_agents):
        px, py = env.agent_positions[i]
        agent_paths[i].append((px, py))
        visit_maps[i][px, py] += 1

    while not done:
        # Action selection
        action, _ = model.predict(obs, deterministic=True)
        action_arr = np.atleast_1d(action)
        actions_taken.append(action_arr.copy())

        # Pre-suppression active frontier tracking
        fire_channel = env.state[0].copy()
        active = fire_channel > env.cfg.spread_threshold
        frontier = np.zeros_like(active, dtype=bool)
        if active.any():
            active_neighbors = convolve(
                active.astype(np.int32), _NEIGHBOR_KERNEL, mode="constant", cval=0
            )
            frontier = active & (active_neighbors < 4)
            total_frontier_cells_count += int(frontier.sum())

        # Apply step
        fire_before = env.state[0].copy()
        obs, _, terminated, truncated, info = env.step(action)
        fire_after = env.state[0].copy()
        fire_sizes.append(float(env.state[0].sum()))

        # Calculate agent-caused suppression per agent (approximate spatial allocation)
        # Check cell suppression around each agent's position
        for i in range(n_agents):
            px, py = env.agent_positions[i]
            agent_paths[i].append((px, py))
            visit_maps[i][px, py] += 1
            
            # Estimate suppression around agent's radius
            r = env.cfg.suppression_radius
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size:
                        delta = fire_before[nx, ny] - fire_after[nx, ny]
                        if delta > 0.01:
                            suppression_maps[i][nx, ny] += delta
                            if frontier[nx, ny]:
                                suppressed_frontier_cells_count += 1

        done = bool(terminated or truncated)

    actions_arr = np.array(actions_taken)
    n_steps = len(actions_arr)

    # 1. Action Entropy and Repetition
    agent_entropies = []
    agent_repeated = []
    for i in range(n_agents):
        act_col = actions_arr[:, i]
        _, counts = np.unique(act_col, return_counts=True)
        probs = counts / n_steps
        ent = -np.sum(probs * np.log2(probs + 1e-12)) if n_steps > 0 else 0.0
        rep = np.sum(act_col[:-1] == act_col[1:]) / max(n_steps - 1, 1)
        agent_entropies.append(ent)
        agent_repeated.append(rep)

    # 2. Overlap Efficiency (Jaccard similarity of path visitations)
    visited_sets = [set(zip(*np.nonzero(vm))) for vm in visit_maps]
    if n_agents > 1:
        pairwise_overlaps = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                inter = len(visited_sets[i].intersection(visited_sets[j]))
                union = len(visited_sets[i].union(visited_sets[j]))
                pairwise_overlaps.append(inter / union if union > 0 else 0.0)
        overlap_efficiency = 1.0 - np.mean(pairwise_overlaps) # 1 = perfect separation
    else:
        overlap_efficiency = 1.0

    # 3. Suppression Partitioning (Cosine similarity of suppression maps)
    if n_agents > 1:
        pairwise_cos = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                flat_i = suppression_maps[i].flatten()
                flat_j = suppression_maps[j].flatten()
                norm_i = np.linalg.norm(flat_i)
                norm_j = np.linalg.norm(flat_j)
                if norm_i > 0 and norm_j > 0:
                    cos = np.dot(flat_i, flat_j) / (norm_i * norm_j)
                else:
                    cos = 0.0
                pairwise_cos.append(cos)
        suppression_partitioning = 1.0 - np.mean(pairwise_cos) # 1 = perfect non-overlapping partitions
    else:
        suppression_partitioning = 1.0

    # 4. Frontier Coverage (ratio of frontier cells successfully suppressed)
    frontier_coverage = (
        suppressed_frontier_cells_count / total_frontier_cells_count
        if total_frontier_cells_count > 0 else 0.0
    )

    # 5. Specialization Emergence (spatial distance between agent centroids)
    if n_agents > 1:
        centroids = []
        for i in range(n_agents):
            xs, ys = zip(*agent_paths[i])
            centroids.append((np.mean(xs), np.mean(ys)))
        pairwise_dists = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                d = np.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
                pairwise_dists.append(d)
        specialization_emergence = float(np.mean(pairwise_dists))
    else:
        specialization_emergence = 0.0

    return {
        "visit_maps": visit_maps,
        "suppression_maps": suppression_maps,
        "agent_paths": agent_paths,
        "mean_entropy": float(np.mean(agent_entropies)),
        "mean_repeated": float(np.mean(agent_repeated)),
        "overlap_efficiency": overlap_efficiency,
        "suppression_partitioning": suppression_partitioning,
        "frontier_coverage": frontier_coverage,
        "specialization_emergence": specialization_emergence,
        "fire_sizes": fire_sizes,
    }


def run_evaluation() -> None:
    """Run V3 evaluation pipeline."""
    cfg = load_config("configs/ppo/marl_v3.yaml")
    regions = [
        {"name": "saudi", "dir": "saudi_eastern_province"},
        {"name": "california", "dir": "california"},
    ]
    agent_counts = [3, 5, 10]
    seeds = [0, 1, 2]

    v3_results_dir = ensure_dir(results_dir() / "v3")
    v3_figs_dir = ensure_dir(Path("figures") / "v3")
    sns.set_theme(style="whitegrid")

    raw_evals = []
    diagnostics = {}

    for reg in regions:
        region_name = reg["name"]
        tensor = np.load(region_tensor_path(reg["dir"], 32))
        logger.info(f"\nEvaluating region: {region_name}")

        for n_agents in agent_counts:
            logger.info(f"  Team size: {n_agents} agents")

            # Config setups
            env_cfg = cfg.env
            env_cfg.reward_mode = "normalized"
            env_cfg.reward_v3.enabled = True

            v3_factory = lambda t=tensor, c=env_cfg, na=n_agents: MultiAgentWildfireEnvV3(
                state_tensor=t, config=c, num_agents=na
            )
            v2_factory = lambda t=tensor, c=env_cfg, na=n_agents: MultiAgentWildfireEnvV2(
                state_tensor=t, config=c, num_agents=na
            )
            v1_factory = lambda t=tensor, c=env_cfg, na=n_agents: MultiAgentWildfireEnv(
                state_tensor=t, config=c, num_agents=na
            )

            sample_env = v3_factory()

            # Baseline policies
            baselines = {
                "noop": NoOpPolicy(sample_env.action_space),
                "random": RandomPolicy(sample_env.action_space, seed=cfg.seed),
                "nearest_fire": NearestFirePolicy(sample_env.action_space, grid_size=32),
                "frontier": FrontierPolicy(sample_env.action_space, grid_size=32),
            }

            for name, policy in baselines.items():
                res = evaluate_policy(
                    policy, v3_factory, n_episodes=20,
                    base_seed=0, deterministic=True, metrics_cfg=cfg.metrics,
                )
                for ep_idx, ep in enumerate(res["episodes"]):
                    raw_evals.append({
                        "region": region_name, "num_agents": n_agents,
                        "policy": name, "version": "baseline",
                        "seed": 0, "episode": ep_idx,
                        "reward": ep["episode_reward"],
                        "burned_cells": ep["burned_cells"],
                        "fire_intensity": ep["fire_intensity"],
                    })

            # Load PPO-V1
            for seed in seeds:
                path = models_dir() / f"ppo_marl_{region_name}_{n_agents}agents_seed_{seed}.zip"
                if path.exists():
                    try:
                        model = load_ppo_model(path, v1_factory())
                        res = evaluate_policy(
                            model, v1_factory, n_episodes=20,
                            base_seed=0, deterministic=True, metrics_cfg=cfg.metrics,
                        )
                        for ep_idx, ep in enumerate(res["episodes"]):
                            raw_evals.append({
                                "region": region_name, "num_agents": n_agents,
                                "policy": "ppo_v1", "version": "v1",
                                "seed": seed, "episode": ep_idx,
                                "reward": ep["episode_reward"],
                                "burned_cells": ep["burned_cells"],
                                "fire_intensity": ep["fire_intensity"],
                            })
                        diagnostics[("v1", region_name, n_agents, seed)] = run_detailed_diagnostics_v3(model, v1_factory, seed)
                    except Exception as e:
                        logger.error(f"Failed to evaluate V1 model at {path}: {e}")

            # Load PPO-V2 (ent_coef = 0.02)
            for seed in seeds:
                path = models_dir() / f"ppo_v2_marl_{region_name}_{n_agents}agents_ent0.02_seed_{seed}.zip"
                if path.exists():
                    try:
                        model = load_ppo_model(path, v2_factory())
                        res = evaluate_policy(
                            model, v2_factory, n_episodes=20,
                            base_seed=0, deterministic=True, metrics_cfg=cfg.metrics,
                        )
                        for ep_idx, ep in enumerate(res["episodes"]):
                            raw_evals.append({
                                "region": region_name, "num_agents": n_agents,
                                "policy": "ppo_v2", "version": "v2",
                                "seed": seed, "episode": ep_idx,
                                "reward": ep["episode_reward"],
                                "burned_cells": ep["burned_cells"],
                                "fire_intensity": ep["fire_intensity"],
                            })
                        diagnostics[("v2", region_name, n_agents, seed)] = run_detailed_diagnostics_v3(model, v2_factory, seed)
                    except Exception as e:
                        logger.error(f"Failed to evaluate V2 model at {path}: {e}")

            # Load PPO-V3
            for seed in seeds:
                path = models_dir() / f"ppo_v3_marl_{region_name}_{n_agents}agents_seed_{seed}.zip"
                if path.exists():
                    try:
                        model = load_ppo_model(path, v3_factory())
                        res = evaluate_policy(
                            model, v3_factory, n_episodes=20,
                            base_seed=0, deterministic=True, metrics_cfg=cfg.metrics,
                        )
                        for ep_idx, ep in enumerate(res["episodes"]):
                            raw_evals.append({
                                "region": region_name, "num_agents": n_agents,
                                "policy": "ppo_v3", "version": "v3",
                                "seed": seed, "episode": ep_idx,
                                "reward": ep["episode_reward"],
                                "burned_cells": ep["burned_cells"],
                                "fire_intensity": ep["fire_intensity"],
                            })
                        diagnostics[("v3", region_name, n_agents, seed)] = run_detailed_diagnostics_v3(model, v3_factory, seed)
                    except Exception as e:
                        logger.error(f"Failed to evaluate V3 model at {path}: {e}")

    df_raw = pd.DataFrame(raw_evals)
    df_raw.to_csv(v3_results_dir / "v3_evaluation_results.csv", index=False)
    logger.info("Saved V3 evaluation results")

    # Save Coordination Metrics Table
    save_coordination_metrics_table(diagnostics, v3_results_dir)

    # Generate Figures
    generate_v3_figures(df_raw, diagnostics, v3_figs_dir)


def save_coordination_metrics_table(diagnostics: dict, results_dir: Path) -> None:
    """Save advanced coordination metrics for all versions."""
    rows = []
    keys = sorted(diagnostics.keys())
    
    # Group by version, region, num_agents
    grouped = {}
    for (version, region, n_agents, seed), diag in diagnostics.items():
        grouped.setdefault((version, region, n_agents), []).append(diag)

    for (version, region, n_agents), diags in grouped.items():
        rows.append({
            "version": version,
            "region": region,
            "num_agents": n_agents,
            "mean_entropy": np.mean([d["mean_entropy"] for d in diags]),
            "mean_repeated_ratio": np.mean([d["mean_repeated"] for d in diags]),
            "overlap_efficiency": np.mean([d["overlap_efficiency"] for d in diags]),
            "suppression_partitioning": np.mean([d["suppression_partitioning"] for d in diags]),
            "frontier_coverage": np.mean([d["frontier_coverage"] for d in diags]),
            "specialization_emergence": np.mean([d["specialization_emergence"] for d in diags]),
        })

    pd.DataFrame(rows).to_csv(results_dir / "marl_coordination_metrics_v3.csv", index=False)
    logger.info("Saved V3 coordination metrics -> marl_coordination_metrics_v3.csv")


def generate_v3_figures(df_raw: pd.DataFrame, diagnostics: dict, figs_dir: Path) -> None:
    """Generate Phase 3 comparative figures."""
    # Figure 1: Policy Comparison on Fire Intensity
    plt.figure(figsize=(10, 6))
    saudi_data = df_raw[df_raw["region"] == "saudi"]
    if not saudi_data.empty:
        policies = ["noop", "nearest_fire", "ppo_v1", "ppo_v2", "ppo_v3"]
        for pol in policies:
            sub = saudi_data[saudi_data["policy"] == pol]
            if sub.empty:
                continue
            agents = sorted(sub["num_agents"].unique())
            means = []
            ci_los, ci_his = [], []
            for n in agents:
                vals = sub[sub["num_agents"] == n]["fire_intensity"].values
                means.append(vals.mean())
                lo, hi = confidence_interval_95(vals)
                ci_los.append(lo)
                ci_his.append(hi)

            plt.plot(agents, means, marker="o", label=pol.upper(), linewidth=2)
            plt.fill_between(agents, ci_los, ci_his, alpha=0.1)

        plt.xlabel("Team Size (Number of Agents)", fontsize=12)
        plt.ylabel("Fire Intensity (Final Sum)", fontsize=12)
        plt.title("MARL Performance Comparison (Saudi Arabia)", fontsize=13, fontweight="bold")
        plt.xticks([3, 5, 10])
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(figs_dir / "marl_performance_comparison_v3.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/v3/marl_performance_comparison_v3.png")

    # Figure 2: Coordination Metrics Radar/Bar comparison for 5-agent Saudi models
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    versions = ["v1", "v2", "v3"]
    metrics = ["overlap_efficiency", "suppression_partitioning", "frontier_coverage", "mean_entropy"]
    
    # Gather metric values for Saudi 5 agents
    bar_data = {m: [] for m in metrics}
    for ver in versions:
        diags = [diagnostics[(ver, "saudi", 5, s)] for s in [0, 1, 2] if (ver, "saudi", 5, s) in diagnostics]
        for m in metrics:
            bar_data[m].append(np.mean([d[m] for d in diags]) if diags else 0.0)

    x = np.arange(len(versions))
    width = 0.2
    for i, m in enumerate(metrics):
        axes[0].bar(x + i*width, bar_data[m], width, label=m.replace("_", " ").title())

    axes[0].set_xlabel("PPO Version", fontsize=12)
    axes[0].set_ylabel("Metric Value", fontsize=12)
    axes[0].set_title("5-Agent Saudi Coordination Diagnostics", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x + width*1.5)
    axes[0].set_xticklabels(["PPO V1", "PPO V2", "PPO V3"])
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # 3. Fire Propagation Trajectory Comparison (Saudi, 5 Agents, Seed=0)
    if ("v1", "saudi", 5, 0) in diagnostics and ("v3", "saudi", 5, 0) in diagnostics:
        v1_fire = diagnostics[("v1", "saudi", 5, 0)]["fire_sizes"]
        v3_fire = diagnostics[("v3", "saudi", 5, 0)]["fire_sizes"]
        axes[1].plot(v1_fire, label="PPO V1 (Collapsed)", color="gray", linestyle="--")
        if ("v2", "saudi", 5, 0) in diagnostics:
            v2_fire = diagnostics[("v2", "saudi", 5, 0)]["fire_sizes"]
            axes[1].plot(v2_fire, label="PPO V2 (Collapsed)", color="orange", linestyle=":")
        axes[1].plot(v3_fire, label="PPO V3 (Stabilized)", color="royalblue", linewidth=2.5)
        
        # Load baseline nearest fire for Saudi 5 agents seed 0
        axes[1].set_xlabel("Simulation Step", fontsize=12)
        axes[1].set_ylabel("Fire Intensity", fontsize=12)
        axes[1].set_title("Trajectory Propagation V1 vs V2 vs V3", fontsize=13, fontweight="bold")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Phase 3 Stabilization Comparison (Saudi, 5 Agents)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figs_dir / "coordination_trajectory_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated: figures/v3/coordination_trajectory_comparison.png")


if __name__ == "__main__":
    run_evaluation()
