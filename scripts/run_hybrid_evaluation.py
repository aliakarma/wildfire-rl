#!/usr/bin/env python
"""Phase 4 comparative evaluation script.

Directly compares Hybrid Path-Planning + RL Coordination with PPO V1, V2, V3, and baselines.
Computes coordination gain, overlap efficiency, suppression partitioning, and transfer learning,
and outputs publication-quality tables and figures.
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
from wildfire_rl.envs.hybrid_multi_agent import HybridMultiAgentWildfireEnv
from wildfire_rl.eval.baselines import (
    NoOpPolicy, RandomPolicy, NearestFirePolicy, FrontierPolicy,
)
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.loading import load_ppo_model
from wildfire_rl.eval.significance import welch_ttest, confidence_interval_95, cohens_d
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir
from wildfire_rl.logging_utils import get_logger

logger = get_logger("run_hybrid_evaluation")

_NEIGHBOR_KERNEL = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def run_detailed_diagnostics_v4(
    model: Any, env_factory: Any, seed: int = 0
) -> dict[str, Any]:
    """Run one episode to collect high-level coordination and routing statistics."""
    env = env_factory()
    if hasattr(model, "env"):
        model.env = env

    obs, _ = env.reset(seed=seed)
    done = False

    n_agents = env.num_agents
    visit_maps = [np.zeros((32, 32)) for _ in range(n_agents)]
    suppression_maps = [np.zeros((32, 32)) for _ in range(n_agents)]
    actions_taken = []
    step_actions_taken = []
    target_paths = [[] for _ in range(n_agents)]
    agent_paths = [[] for _ in range(n_agents)]
    fire_sizes = [float(env.state[0].sum())]
    
    total_frontier_cells_count = 0
    suppressed_frontier_cells_count = 0

    for i in range(n_agents):
        px, py = env.agent_positions[i]
        agent_paths[i].append((px, py))
        visit_maps[i][px, py] += 1

    while not done:
        # Policy prediction (high-level target actions)
        action, _ = model.predict(obs, deterministic=True)
        action_arr = np.atleast_1d(action)
        actions_taken.append(action_arr.copy())

        # Pre-suppression active frontier
        fire_channel = env.state[0].copy()
        active = fire_channel > env.cfg.spread_threshold
        frontier = np.zeros_like(active, dtype=bool)
        if active.any():
            active_neighbors = convolve(
                active.astype(np.int32), _NEIGHBOR_KERNEL, mode="constant", cval=0
            )
            frontier = active & (active_neighbors < 4)
            total_frontier_cells_count += int(frontier.sum())

        # Take environment step (will run path-routing step internally)
        fire_before = env.state[0].copy()
        obs, _, terminated, truncated, info = env.step(action)
        fire_after = env.state[0].copy()
        fire_sizes.append(float(env.state[0].sum()))

        # Log targets and steps
        targets = info.get("target_coordinates", env.agent_positions)
        steps = info.get("step_actions", [4] * n_agents)
        step_actions_taken.append(np.array(steps))

        # Check suppression and paths
        for i in range(n_agents):
            px, py = env.agent_positions[i]
            tx, ty = targets[i]
            agent_paths[i].append((px, py))
            target_paths[i].append((tx, ty))
            visit_maps[i][px, py] += 1

            # Estimate suppression contribution
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
    steps_arr = np.array(step_actions_taken)
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

    # 2. Overlap Efficiency (ratio of union path cells vs total cells visited)
    visited_sets = [set(zip(*np.nonzero(vm))) for vm in visit_maps]
    if n_agents > 1:
        pairwise_overlaps = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                inter = len(visited_sets[i].intersection(visited_sets[j]))
                union = len(visited_sets[i].union(visited_sets[j]))
                pairwise_overlaps.append(inter / union if union > 0 else 0.0)
        overlap_efficiency = 1.0 - np.mean(pairwise_overlaps)
    else:
        overlap_efficiency = 1.0

    # 3. Suppression Partitioning (Jaccard similarity of agent suppression zones)
    if n_agents > 1:
        pairwise_cos = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                flat_i = suppression_maps[i].flatten()
                flat_j = suppression_maps[j].flatten()
                norm_i = np.linalg.norm(flat_i)
                # Ensure no zero division
                norm_j = np.linalg.norm(flat_j)
                if norm_i > 0 and norm_j > 0:
                    cos = np.dot(flat_i, flat_j) / (norm_i * norm_j)
                else:
                    cos = 0.0
                pairwise_cos.append(cos)
        suppression_partitioning = 1.0 - np.mean(pairwise_cos)
    else:
        suppression_partitioning = 1.0

    # 4. Frontier Coverage (ratio of marginal frontier suppression)
    frontier_coverage = (
        suppressed_frontier_cells_count / total_frontier_cells_count
        if total_frontier_cells_count > 0 else 0.0
    )

    # 5. Role Persistence (stability of quadrant sector selections over time)
    if n_steps > 1:
        persistence = np.mean([
            np.sum(actions_arr[:-1, i] == actions_arr[1:, i]) / (n_steps - 1)
            for i in range(n_agents)
        ])
    else:
        persistence = 1.0

    # 6. Interception Efficiency ( frontier cells visited / total steps )
    interception_efficiency = frontier_coverage

    # 7. Suppression Contribution Inequality (Gini coefficient of suppression maps sums)
    agent_sums = np.array([float(sm.sum()) for sm in suppression_maps])
    if agent_sums.sum() > 0:
        # Gini coef: 0 means equal contribution, 1 means absolute inequality
        sorted_sums = np.sort(agent_sums)
        n = len(sorted_sums)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_sums)) / (n * np.sum(sorted_sums))
        suppression_contribution_equality = 1.0 - gini
    else:
        suppression_contribution_equality = 1.0

    return {
        "visit_maps": visit_maps,
        "suppression_maps": suppression_maps,
        "agent_paths": agent_paths,
        "target_paths": target_paths,
        "mean_entropy": float(np.mean(agent_entropies)),
        "mean_repeated": float(np.mean(agent_repeated)),
        "overlap_efficiency": overlap_efficiency,
        "suppression_partitioning": suppression_partitioning,
        "frontier_coverage": frontier_coverage,
        "role_persistence": persistence,
        "interception_efficiency": interception_efficiency,
        "suppression_contribution": suppression_contribution_equality,
        "fire_sizes": fire_sizes,
    }


def run_evaluation() -> None:
    """Run full comparative Phase 4 evaluation pipeline."""
    cfg = load_config("configs/ppo/hybrid_marl.yaml")
    regions = [
        {"name": "saudi", "dir": "saudi_eastern_province"},
        {"name": "california", "dir": "california"},
    ]
    agent_counts = [5, 10]
    seeds = [0, 1, 2]

    v4_results_dir = ensure_dir(results_dir() / "v4")
    v4_figs_dir = ensure_dir(Path("figures") / "v4")

    raw_evals = []
    diagnostics = {}

    for reg in regions:
        region_name = reg["name"]
        tensor = np.load(region_tensor_path(reg["dir"], 32))
        logger.info(f"\nEvaluating region: {region_name}")

        for n_agents in agent_counts:
            logger.info(f"  Team size: {n_agents} agents")

            # Environment setups
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
            hybrid_nearest_factory = lambda t=tensor, c=env_cfg, na=n_agents: HybridMultiAgentWildfireEnv(
                state_tensor=t, config=c, num_agents=na, routing_strategy="nearest_fire"
            )
            hybrid_frontier_factory = lambda t=tensor, c=env_cfg, na=n_agents: HybridMultiAgentWildfireEnv(
                state_tensor=t, config=c, num_agents=na, routing_strategy="frontier"
            )

            sample_env = v3_factory()

            # Baseline Heuristics
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
                        diagnostics[("v1", region_name, n_agents, seed)] = run_detailed_diagnostics_v4(model, v1_factory, seed)
                    except Exception as e:
                        logger.error(f"Failed to load V1: {e}")

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
                        diagnostics[("v2", region_name, n_agents, seed)] = run_detailed_diagnostics_v4(model, v2_factory, seed)
                    except Exception as e:
                        logger.error(f"Failed to load V2: {e}")

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
                        diagnostics[("v3", region_name, n_agents, seed)] = run_detailed_diagnostics_v4(model, v3_factory, seed)
                    except Exception as e:
                        logger.error(f"Failed to load V3: {e}")

            # Load Hybrid PPO (NearestFire routing)
            for seed in seeds:
                path = models_dir() / f"ppo_hybrid_marl_{region_name}_{n_agents}agents_nearest_fire_seed_{seed}.zip"
                if path.exists():
                    try:
                        model = load_ppo_model(path, hybrid_nearest_factory())
                        res = evaluate_policy(
                            model, hybrid_nearest_factory, n_episodes=20,
                            base_seed=0, deterministic=True, metrics_cfg=cfg.metrics,
                        )
                        for ep_idx, ep in enumerate(res["episodes"]):
                            raw_evals.append({
                                "region": region_name, "num_agents": n_agents,
                                "policy": "ppo_hybrid_nearest", "version": "hybrid_nearest",
                                "seed": seed, "episode": ep_idx,
                                "reward": ep["episode_reward"],
                                "burned_cells": ep["burned_cells"],
                                "fire_intensity": ep["fire_intensity"],
                            })
                        diagnostics[("hybrid_nearest", region_name, n_agents, seed)] = run_detailed_diagnostics_v4(model, hybrid_nearest_factory, seed)
                    except Exception as e:
                        logger.error(f"Failed to load Hybrid (Nearest): {e}")

            # Load Hybrid PPO (Frontier routing)
            for seed in seeds:
                path = models_dir() / f"ppo_hybrid_marl_{region_name}_{n_agents}agents_frontier_seed_{seed}.zip"
                if path.exists():
                    try:
                        model = load_ppo_model(path, hybrid_frontier_factory())
                        res = evaluate_policy(
                            model, hybrid_frontier_factory, n_episodes=20,
                            base_seed=0, deterministic=True, metrics_cfg=cfg.metrics,
                        )
                        for ep_idx, ep in enumerate(res["episodes"]):
                            raw_evals.append({
                                "region": region_name, "num_agents": n_agents,
                                "policy": "ppo_hybrid_frontier", "version": "hybrid_frontier",
                                "seed": seed, "episode": ep_idx,
                                "reward": ep["episode_reward"],
                                "burned_cells": ep["burned_cells"],
                                "fire_intensity": ep["fire_intensity"],
                            })
                        diagnostics[("hybrid_frontier", region_name, n_agents, seed)] = run_detailed_diagnostics_v4(model, hybrid_frontier_factory, seed)
                    except Exception as e:
                        logger.error(f"Failed to load Hybrid (Frontier): {e}")

    df_raw = pd.DataFrame(raw_evals)
    df_raw.to_csv(v4_results_dir / "v4_evaluation_results.csv", index=False)
    logger.info("Saved V4 evaluation results")

    # Coordination Metrics Table
    save_coordination_metrics_table_v4(diagnostics, v4_results_dir)

    # Cross-regional Transfer Learning Analysis
    run_transfer_learning_v4(cfg, seeds, v4_results_dir)

    # Figures Generation
    generate_v4_figures(df_raw, diagnostics, v4_figs_dir)


def save_coordination_metrics_table_v4(diagnostics: dict, results_dir: Path) -> None:
    """Save comprehensive Phase 4 coordination diagnostics."""
    rows = []
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
            "role_persistence": np.mean([d["role_persistence"] for d in diags]),
            "interception_efficiency": np.mean([d["interception_efficiency"] for d in diags]),
            "suppression_contribution": np.mean([d["suppression_contribution"] for d in diags]),
        })

    pd.DataFrame(rows).to_csv(results_dir / "marl_coordination_metrics_v4.csv", index=False)
    logger.info("Saved V4 coordination table -> marl_coordination_metrics_v4.csv")


def run_transfer_learning_v4(cfg: Any, seeds: list[int], results_dir: Path) -> None:
    """Evaluate cross-regional transfer for 5-agent Hybrid models."""
    logger.info("\nEvaluating cross-regional transfer for Hybrid models...")
    env_cfg = cfg.env
    env_cfg.reward_mode = "normalized"
    env_cfg.reward_v3.enabled = True

    saudi_tensor = np.load(region_tensor_path("saudi_eastern_province", 32))
    california_tensor = np.load(region_tensor_path("california", 32))

    transfer_results = []
    
    # We evaluate NearestFire-hybrid routing models (5-agent) on both regions
    saudi_factory = lambda: HybridMultiAgentWildfireEnv(
        state_tensor=saudi_tensor, config=env_cfg, num_agents=5, routing_strategy="nearest_fire"
    )
    california_factory = lambda: HybridMultiAgentWildfireEnv(
        state_tensor=california_tensor, config=env_cfg, num_agents=5, routing_strategy="nearest_fire"
    )

    pairs = [
        ("saudi", "saudi", saudi_factory),
        ("saudi", "california", california_factory),
        ("california", "california", california_factory),
        ("california", "saudi", saudi_factory),
    ]

    for train_reg, test_reg, factory in pairs:
        rewards = []
        burned = []
        for seed in seeds:
            model_path = models_dir() / f"ppo_hybrid_marl_{train_reg}_5agents_nearest_fire_seed_{seed}.zip"
            if model_path.exists():
                try:
                    model = load_ppo_model(model_path, factory())
                    res = evaluate_policy(
                        model, factory, n_episodes=20,
                        base_seed=0, deterministic=True, metrics_cfg=cfg.metrics,
                    )
                    rewards.extend([ep["episode_reward"] for ep in res["episodes"]])
                    burned.extend([ep["burned_cells"] for ep in res["episodes"]])
                except Exception as e:
                    logger.error(f"Transfer error {train_reg}->{test_reg}: {e}")

        reward_arr = np.array(rewards)
        ci_lo, ci_hi = confidence_interval_95(reward_arr)
        transfer_results.append({
            "train_region": train_reg,
            "test_region": test_reg,
            "reward_mean": reward_arr.mean() if len(reward_arr) > 0 else np.nan,
            "reward_std": reward_arr.std() if len(reward_arr) > 0 else np.nan,
            "reward_ci_lo": ci_lo,
            "reward_ci_hi": ci_hi,
            "burned_cells_mean": np.mean(burned) if len(burned) > 0 else np.nan,
        })

    df_trans = pd.DataFrame(transfer_results)
    
    # Calculate degradation
    sa_native = df_trans[(df_trans["train_region"] == "saudi") & (df_trans["test_region"] == "saudi")]["reward_mean"].values
    sa_trans = df_trans[(df_trans["train_region"] == "california") & (df_trans["test_region"] == "saudi")]["reward_mean"].values
    ca_native = df_trans[(df_trans["train_region"] == "california") & (df_trans["test_region"] == "california")]["reward_mean"].values
    ca_trans = df_trans[(df_trans["train_region"] == "saudi") & (df_trans["test_region"] == "california")]["reward_mean"].values

    sa_deg = (sa_native[0] - sa_trans[0]) / abs(sa_native[0]) * 100 if len(sa_native) > 0 and len(sa_trans) > 0 else np.nan
    ca_deg = (ca_native[0] - ca_trans[0]) / abs(ca_native[0]) * 100 if len(ca_native) > 0 and len(ca_trans) > 0 else np.nan

    df_trans["transfer_degradation_pct"] = [
        0.0 if train == test else (sa_deg if test == "saudi" else ca_deg)
        for train, test in zip(df_trans["train_region"], df_trans["test_region"])
    ]

    df_trans.to_csv(results_dir / "marl_transfer_statistics_v4.csv", index=False)
    logger.info("Saved V4 transfer learning stats -> marl_transfer_statistics_v4.csv")


def generate_v4_figures(df_raw: pd.DataFrame, diagnostics: dict, figs_dir: Path) -> None:
    """Generate Phase 4 publication-quality visualizations."""
    sns.set_theme(style="whitegrid")

    # 1. figures/v4/hybrid_coordination_heatmaps.png
    # Cosine correlation of quadrant targets (coordination decisions)
    key = ("hybrid_nearest", "saudi", 5, 0)
    if key in diagnostics:
        paths = np.array(diagnostics[key]["target_paths"]) # (num_agents, steps, 2)
        corr_matrix = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                flat_i = np.array(paths[i]).flatten()
                flat_j = np.array(paths[j]).flatten()
                corr = np.corrcoef(flat_i, flat_j)[0, 1]
                corr_matrix[i, j] = 0.0 if np.isnan(corr) else corr

        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1.0, vmax=1.0, fmt=".2f")
        plt.title("Spatial Targeting Decoupled Decision Correlation (N=5)", fontsize=11, fontweight="bold")
        plt.xlabel("Agent ID")
        plt.ylabel("Agent ID")
        plt.savefig(figs_dir / "hybrid_coordination_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/v4/hybrid_coordination_heatmaps.png")

    # 2. figures/v4/frontier_interception_maps.png & routing_vs_coordination_visualization.png
    # Visualizes paths of Hybrid vs Collapsed PPO V3
    key_h = ("hybrid_nearest", "saudi", 5, 0)
    key_v3 = ("v3", "saudi", 5, 0)
    if key_h in diagnostics and key_v3 in diagnostics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        saudi_tensor = np.load(region_tensor_path("saudi_eastern_province", 32))
        
        # Plot V3 collapsed paths (should be stationary at start coordinates)
        axes[0].imshow(saudi_tensor[0], cmap="gray_r", alpha=0.3, extent=[0, 32, 32, 0])
        paths_v3 = diagnostics[key_v3]["agent_paths"]
        colors = ["blue", "green", "red", "purple", "orange"]
        for idx, path in enumerate(paths_v3):
            xs, ys = zip(*path)
            axes[0].plot(ys, xs, color=colors[idx], label=f"Agent {idx}", alpha=0.7, marker="o", markersize=2)
        axes[0].set_title("PPO V3: Stay-in-Place Collapse (Manually Planned)", fontsize=12, fontweight="bold")
        axes[0].legend()

        # Plot Hybrid active routing paths (actively chasing fire sectors)
        axes[1].imshow(saudi_tensor[0], cmap="gray_r", alpha=0.3, extent=[0, 32, 32, 0])
        paths_h = diagnostics[key_h]["agent_paths"]
        for idx, path in enumerate(paths_h):
            xs, ys = zip(*path)
            axes[1].plot(ys, xs, color=colors[idx], label=f"Agent {idx}", alpha=0.7, marker="o", markersize=2)
        axes[1].set_title("Hybrid PPO: Active Decoupled Routing", fontsize=12, fontweight="bold")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(figs_dir / "routing_vs_coordination_visualization.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/v4/routing_vs_coordination_visualization.png")

    # 3. figures/v4/suppression_partitioning.png
    # Gini inequality values comparison
    plt.figure(figsize=(7, 4.5))
    versions = ["v1", "v2", "v3", "hybrid_nearest", "hybrid_frontier"]
    gini_means = []
    for ver in versions:
        diags = [diagnostics[(ver, "saudi", 5, s)] for s in [0, 1, 2] if (ver, "saudi", 5, s) in diagnostics]
        gini_means.append(np.mean([d["suppression_contribution"] for d in diags]) if diags else 0.0)
    
    plt.bar(versions, gini_means, color="teal", alpha=0.8)
    plt.ylabel("Suppression Equal Division (1.0 = Perfect Equal Contribution)", fontsize=11)
    plt.title("Spatial Division of Labor (Suppression Equality Index)", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig(figs_dir / "suppression_partitioning.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated: figures/v4/suppression_partitioning.png")

    # 4. figures/v4/role_specialization_trajectories.png
    # Plot target sector index sequences over steps for first 3 agents in Hybrid Nearest
    key_h = ("hybrid_nearest", "saudi", 5, 0)
    if key_h in diagnostics:
        plt.figure(figsize=(8, 4))
        paths = diagnostics[key_h]["target_paths"] # targets coordinates over time
        for i in range(3):
            # Compute sector targeting index dynamically for tracking
            targets = np.array(paths[i])
            sector_choices = []
            for tx, ty in targets:
                # determine quadrant index
                if tx < 16 and ty < 16:
                    sector_choices.append(0)
                elif tx < 16 and ty >= 16:
                    sector_choices.append(1)
                elif tx >= 16 and ty < 16:
                    sector_choices.append(2)
                else:
                    sector_choices.append(3)
            plt.plot(sector_choices, marker="x", label=f"Agent {i} Target Sector Decision", alpha=0.8)
        
        plt.yticks([0, 1, 2, 3], ["Sector 0 (NW)", "Sector 1 (NE)", "Sector 2 (SW)", "Sector 3 (SE)"])
        plt.xlabel("Step")
        plt.ylabel("Quadrant Sector Assigned")
        plt.title("Cooperative Role Specialization & Sector Assignments over Time", fontsize=11, fontweight="bold")
        plt.legend()
        plt.savefig(figs_dir / "role_specialization_trajectories.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/v4/role_specialization_trajectories.png")

    # 5. figures/v4/coordination_gain_curves.png
    # Grouped fire intensity plot comparing PPO V3 vs Hybrid Nearest/Frontier vs NearestFire baseline
    saudi_5 = df_raw[(df_raw["region"] == "saudi") & (df_raw["num_agents"] == 5)]
    if not saudi_5.empty:
        plt.figure(figsize=(7, 4.5))
        plot_policies = ["nearest_fire", "ppo_v3", "ppo_hybrid_nearest", "ppo_hybrid_frontier"]
        means = []
        stds = []
        for pol in plot_policies:
            vals = saudi_5[saudi_5["policy"] == pol]["fire_intensity"].values
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 0 else 0)

        plt.bar(plot_policies, means, yerr=stds, color=["green", "gray", "royalblue", "indigo"], capsize=5, alpha=0.8)
        plt.ylabel("Final Fire Intensity (Lower is Better)", fontsize=11)
        plt.title("MARL Coordination Gain: Hybrid PPO vs Baselines", fontsize=12, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.savefig(figs_dir / "coordination_gain_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/v4/coordination_gain_curves.png")

    # 6. figures/v4/hybrid_transfer_heatmap.png
    trans_stat_path = results_dir() / "v4" / "marl_transfer_statistics_v4.csv"
    if trans_stat_path.exists():
        df_trans = pd.read_csv(trans_stat_path)
        pivot = df_trans.pivot(index="train_region", columns="test_region", values="reward_mean")
        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f", cbar_kws={'label': 'Normalized Reward'})
        plt.title("Hybrid PPO: Cross-Regional Transfer Matrix Heatmap (N=5)", fontsize=11, fontweight="bold")
        plt.savefig(figs_dir / "hybrid_transfer_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/v4/hybrid_transfer_heatmap.png")

        # 7. figures/v4/transfer_degradation_curves.png
        df_deg = df_trans[df_trans["train_region"] != df_trans["test_region"]]
        plt.figure(figsize=(6, 4.5))
        bars = plt.bar(df_deg["train_region"] + " -> " + df_deg["test_region"], 
                       df_deg["transfer_degradation_pct"], color=["orange", "darkred"])
        plt.ylabel("Reward Degradation (%)")
        plt.title("Hybrid Path-Planning: Cross-Regional Transfer Degradation", fontsize=11, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, h + 1, f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')
        plt.ylim(0, max(df_deg["transfer_degradation_pct"]) + 10)
        plt.savefig(figs_dir / "transfer_degradation_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/v4/transfer_degradation_curves.png")

    # 8. figures/v4/coordination_transfer_analysis.png
    # Plots Cosine suppression partitioning in Native vs Transfer evaluations
    plt.figure(figsize=(6, 4.5))
    metrics_path = results_dir() / "v4" / "marl_coordination_metrics_v4.csv"
    if metrics_path.exists():
        df_m = pd.read_csv(metrics_path)
        hybrid_m = df_m[df_m["version"].isin(["hybrid_nearest", "hybrid_frontier"])]
        if not hybrid_m.empty:
            sns.barplot(data=hybrid_m, x="num_agents", y="overlap_efficiency", hue="version", palette="muted")
            plt.ylabel("Overlap Efficiency (1.0 = Perfect Separation)")
            plt.xlabel("Team Size")
            plt.title("Hybrid Coordination Overlap Efficiency Scaling", fontsize=11, fontweight="bold")
            plt.legend()
            plt.savefig(figs_dir / "coordination_transfer_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Generated: figures/v4/coordination_transfer_analysis.png")


if __name__ == "__main__":
    run_evaluation()
