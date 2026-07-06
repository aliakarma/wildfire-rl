#!/usr/bin/env python
"""Controllability scientific validation driver (Phase 1.5).

Evaluates NoOp, Random, NearestFire, Frontier, and PPO on Saudi & California.
Computes Welch's t-test, Cohen's d, action entropy, repeated-action ratios,
spatial coverage, and intervention impact, and outputs 4 figures and 3 tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from wildfire_rl.config import load_config
from wildfire_rl.envs.base import make_env_factory
from wildfire_rl.eval.baselines import FrontierPolicy, NearestFirePolicy, NoOpPolicy, RandomPolicy
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.loading import load_ppo_model
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir

# Action mappings: 0 up, 1 down, 2 left, 3 right, 4 stay
STAY_ACTION = 4


def _find_model(region_name: str, grid: int, seed: int = 0) -> Path | None:
    candidates = [
        models_dir() / f"ppo_{region_name}_{grid}_seed_{seed}.zip",
        models_dir() / f"ppo_{region_name}_{grid}x{grid}.zip",
        models_dir() / f"ppo_{region_name}_{grid}x{grid}_100k_seed_{seed}.zip",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def run_scientific_validation() -> None:
    cfg = load_config()
    regions = [
        {"name": "saudi", "dir": "saudi_eastern_province", "grid": 32},
        {"name": "california", "dir": "california", "grid": 32},
    ]

    stats_rows = []
    effect_rows = []
    action_influence_rows = []

    # Store trajectories for plotting (we'll capture Saudi for visualization)
    all_trajectories = {}
    all_influence_maps = {}
    all_entropies = {}

    for reg in regions:
        region_name = reg["name"]
        print(f"\n==================== Region: {region_name} ====================")
        tensor_path = region_tensor_path(reg["dir"], reg["grid"])
        tensor = np.load(tensor_path)

        # Override config region
        cfg.region.name = region_name
        cfg.region.dir = reg["dir"]
        cfg.region.grid_size = reg["grid"]

        factory = make_env_factory(state_tensor=tensor, config=cfg.env)
        env = factory()

        policies = {
            "noop": NoOpPolicy(env.action_space),
            "random": RandomPolicy(env.action_space, seed=0),
            "nearest_fire": NearestFirePolicy(env.action_space, grid_size=reg["grid"], env=env),
            "frontier": FrontierPolicy(env.action_space, grid_size=reg["grid"], env=env),
        }

        model_path = _find_model(region_name, reg["grid"], seed=0)
        if model_path is not None:
            policies["ppo"] = load_ppo_model(model_path, env)
        else:
            print(f"Warning: No PPO checkpoint found for {region_name}")

        episode_rewards = {}
        episode_burned = {}
        episode_fire = {}

        # Trajectory diagnostics tracking
        for name, policy in policies.items():
            print(f"Running {name}...")
            result = evaluate_policy(
                policy,
                factory,
                n_episodes=cfg.eval.n_episodes,
                base_seed=cfg.eval.base_seed,
                deterministic=cfg.eval.deterministic,
                metrics_cfg=cfg.metrics,
            )
            s = result["summary"]

            # Extract raw values for statistical tests
            rewards = np.array([ep["episode_reward"] for ep in result["episodes"]])
            burned = np.array([ep["burned_cells"] for ep in result["episodes"]])
            fire = np.array([ep["fire_intensity"] for ep in result["episodes"]])

            episode_rewards[name] = rewards
            episode_burned[name] = burned
            episode_fire[name] = fire

            # Record baseline stats row
            stats_rows.append(
                {
                    "region": region_name,
                    "policy": name,
                    "reward_mean": s.get("episode_reward_mean"),
                    "reward_std": s.get("episode_reward_std"),
                    "reward_ci_lo": s.get("episode_reward_mean")
                    - 1.96 * s.get("episode_reward_std") / np.sqrt(len(rewards)),
                    "reward_ci_hi": s.get("episode_reward_mean")
                    + 1.96 * s.get("episode_reward_std") / np.sqrt(len(rewards)),
                    "burned_cells_mean": s.get("burned_cells_mean"),
                    "burned_cells_std": s.get("burned_cells_std"),
                    "fire_intensity_mean": s.get("fire_intensity_mean"),
                    "fire_intensity_std": s.get("fire_intensity_std"),
                }
            )

            # Run 1 detailed trajectory evaluation to measure collapse, action entropy, etc.
            sample_env = factory()
            obs, _ = sample_env.reset(seed=cfg.eval.base_seed)
            done = False
            actions = []
            fire_sizes = []
            agent_visits = np.zeros((reg["grid"], reg["grid"]))

            # We track initial fire total to calculate relative containment metrics
            init_fire = float(sample_env.state[0].sum()) or 1.0

            while not done:
                action, _ = policy.predict(obs, deterministic=cfg.eval.deterministic)
                actions.append(int(action))
                obs, _, terminated, truncated, _ = sample_env.step(action)
                fire_sizes.append(float(sample_env.state[0].sum()))

                ax, ay = sample_env.agent_pos
                agent_visits[ax, ay] += 1
                done = bool(terminated or truncated)

            # Compute Policy Collapse Metrics
            actions = np.array(actions)
            n_steps = len(actions)

            # Entropy: -sum(p * log(p))
            _, counts = np.unique(actions, return_counts=True)
            probs = counts / n_steps
            entropy = -np.sum(probs * np.log2(probs)) if n_steps > 0 else 0.0

            # Action diversity: ratio of unique actions taken
            diversity = (
                len(counts) / len(env.action_space.nvec)
                if hasattr(env.action_space, "nvec")
                else len(counts) / 5
            )

            # Repeated action ratio: fraction of consecutive same actions
            consecutive = (
                np.sum(actions[:-1] == actions[1:]) / (n_steps - 1) if n_steps > 1 else 0.0
            )

            # Spatial coverage: fraction of unique grid cells visited
            unique_visited = np.count_nonzero(agent_visits)
            spatial_coverage = unique_visited / (reg["grid"] ** 2)

            # Containment efficiency: fraction of final fire to initial fire
            final_fire = fire_sizes[-1]
            containment_eff = 1.0 - (final_fire / init_fire)

            # Spread rate: average step change in fire sum
            diffs = np.diff(fire_sizes)
            spread_rate = np.mean(diffs) if len(diffs) > 0 else 0.0

            # Extinguished cells sum
            # Suppression radius is 1, so the agent acts on 3x3 patch
            extinguished_cells = (init_fire - final_fire) if final_fire < init_fire else 0.0

            action_influence_rows.append(
                {
                    "region": region_name,
                    "policy": name,
                    "action_entropy": entropy,
                    "action_diversity": diversity,
                    "repeated_action_ratio": consecutive,
                    "spatial_coverage": spatial_coverage,
                    "containment_efficiency": containment_eff,
                    "spread_rate": spread_rate,
                    "extinguished_cells": extinguished_cells,
                }
            )

            # Save Saudi trajectories/heatmaps for plotting
            if region_name == "saudi":
                all_trajectories[name] = fire_sizes
                all_influence_maps[name] = agent_visits
                all_entropies[name] = entropy

        # Welch's T-Test and Cohen's d against baseline policies (PPO vs NoOp and PPO vs Random)
        if "ppo" in episode_rewards:
            ppo_rewards = episode_rewards["ppo"]
            for comp_name in ["noop", "random", "nearest_fire", "frontier"]:
                if comp_name in episode_rewards:
                    comp_rewards = episode_rewards[comp_name]
                    t_res = ttest_ind(ppo_rewards, comp_rewards, equal_var=False)
                    d = compute_cohens_d(ppo_rewards, comp_rewards)
                    effect_rows.append(
                        {
                            "region": region_name,
                            "comparison": f"ppo_vs_{comp_name}",
                            "p_value": t_res.pvalue,
                            "cohens_d": d,
                            "significant": bool(t_res.pvalue < 0.05),
                        }
                    )

    # Save Tables
    stats_df = pd.DataFrame(stats_rows)
    stats_out = ensure_dir(results_dir()) / "baseline_statistics.csv"
    stats_df.to_csv(stats_out, index=False)
    print(f"Wrote baseline statistics -> {stats_out}")

    effects_df = pd.DataFrame(effect_rows)
    effects_out = ensure_dir(results_dir()) / "effect_sizes.csv"
    effects_df.to_csv(effects_out, index=False)
    print(f"Wrote effect sizes -> {effects_out}")

    influence_df = pd.DataFrame(action_influence_rows)
    influence_out = ensure_dir(results_dir()) / "action_influence_metrics.csv"
    influence_df.to_csv(influence_out, index=False)
    print(f"Wrote action influence metrics -> {influence_out}")

    # Generate Figures
    figs_dir = Path("figures")
    figs_dir.mkdir(exist_ok=True)

    # 1. Action Entropy Curves
    plt.figure(figsize=(7, 4.5))
    names = list(all_entropies.keys())
    entropies = list(all_entropies.values())
    plt.bar(names, entropies, color=["gray", "blue", "green", "purple", "red"][: len(names)])
    plt.ylabel("Action Entropy (Bits)")
    plt.title("Action Entropy across Policies")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    entropy_out = figs_dir / "action_entropy_curves.png"
    plt.savefig(entropy_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote action entropy curves -> {entropy_out}")

    # 2. Policy Action Heatmaps (Visualizing suppression trajectory density)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    plot_names = ["noop", "random", "nearest_fire", "ppo"]
    for idx, name in enumerate(plot_names):
        if name in all_influence_maps:
            visits = all_influence_maps[name]
            im = axes[idx].imshow(visits, cmap="hot", interpolation="nearest")
            axes[idx].set_title(f"Visits density: {name}")
            fig.colorbar(im, ax=axes[idx])
        else:
            axes[idx].axis("off")
    heatmap_out = figs_dir / "policy_action_heatmaps.png"
    plt.savefig(heatmap_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote policy action heatmaps -> {heatmap_out}")

    # 3. Trajectory Diversity Plot
    plt.figure(figsize=(8, 5))
    for name, path in all_trajectories.items():
        plt.plot(path, label=f"Trajectory: {name}", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Fire Intensity (Sum)")
    plt.title("Stochastic Trajectory Diversity (Saudi)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    traj_div_out = figs_dir / "trajectory_diversity.png"
    plt.savefig(traj_div_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote trajectory diversity plot -> {traj_div_out}")

    # 4. Policy Behavior Comparison Overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, path in all_trajectories.items():
        steps = np.arange(len(path))
        ax.plot(steps, path, label=f"{name} behavior", linewidth=2.5)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Fire Area Sum")
    ax.set_title("Wildfire Evolution Comparison under Baseline Intervention")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    behavior_out = figs_dir / "policy_behavior_comparison.png"
    plt.savefig(behavior_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote policy behavior comparison -> {behavior_out}")

    # 5. Intervention Impact Curves
    plt.figure(figsize=(8, 5))
    noop_traj = np.array(all_trajectories["noop"])
    for name, path in all_trajectories.items():
        if name == "noop":
            continue
        min_len = min(len(noop_traj), len(path))
        impact = noop_traj[:min_len] - np.array(path[:min_len])
        plt.plot(impact, label=f"Impact: {name} vs Noop", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Extinguished Grid Area")
    plt.title("Intervention Fire Growth Mitigation Over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    impact_curves_out = figs_dir / "intervention_impact_curves.png"
    plt.savefig(impact_curves_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote intervention impact curves -> {impact_curves_out}")


if __name__ == "__main__":
    run_scientific_validation()
