#!/usr/bin/env python
"""Controllability and environment audit analysis (Phase 1).

Measures suppression coverage, runs baseline policies, quantifies action influence,
and writes publication-quality figures and tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wildfire_rl.config import load_config
from wildfire_rl.envs.base import make_env_factory
from wildfire_rl.eval.baselines import NoOpPolicy, RandomPolicy, NearestFirePolicy, FrontierPolicy
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir

# Actions mapping: 0 up, 1 down, 2 left, 3 right, 4 stay (extinguish)
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


def run_controllability(
    config_path: str | None = None,
    overrides: list[str] | None = None,
    seed: int = 0,
) -> None:
    cfg = load_config(config_path, overrides)
    tensor_path = region_tensor_path(cfg.region.dir, cfg.region.grid_size)
    tensor = np.load(tensor_path)
    
    # 1. Environment Audit Calculations
    grid_size = cfg.region.grid_size
    suppression_radius = cfg.env.suppression_radius
    cells_suppressed = (2 * suppression_radius + 1) ** 2
    total_cells = grid_size ** 2
    coverage_ratio = cells_suppressed / total_cells
    
    audit_stats = {
        "grid_size": grid_size,
        "suppression_radius": suppression_radius,
        "cells_suppressed_per_action": cells_suppressed,
        "total_cells": total_cells,
        "suppression_coverage_ratio": coverage_ratio,
        "base_spread": cfg.env.base_spread,
        "decay_rate": cfg.env.decay,
        "fuel_coeff": cfg.env.fuel_coeff,
        "wind_coeff": cfg.env.wind_coeff,
        "terrain_coeff": cfg.env.terrain_coeff,
    }
    
    # Save suppression coverage stats
    stats_out = ensure_dir(results_dir()) / "suppression_coverage_statistics.txt"
    with open(stats_out, "w") as f:
        json.dump(audit_stats, f, indent=2)
    print(f"Wrote suppression coverage stats -> {stats_out}")

    # Build factories and policies
    factory = make_env_factory(state_tensor=tensor, config=cfg.env)
    env = factory()

    policies = {
        "noop": NoOpPolicy(env.action_space),
        "random": RandomPolicy(env.action_space, seed=seed),
        "nearest_fire": NearestFirePolicy(env.action_space, grid_size=grid_size, env=env),
        "frontier": FrontierPolicy(env.action_space, grid_size=grid_size, env=env),
    }

    # Search for PPO Saudi model
    model_path = _find_model(cfg.region.name, cfg.region.grid_size, seed)
    if model_path is not None:
        from wildfire_rl.eval.loading import load_ppo_model
        print(f"Loading trained PPO model from {model_path} for comparison")
        policies["ppo"] = load_ppo_model(model_path, env)
    else:
        print("Warning: No PPO checkpoint found. Baseline evaluation will focus on heuristics.")

    # 2. Controllability Experiments: Run and save metrics
    rows = []
    trajectories = {}
    influence_maps = {}

    for name, policy in policies.items():
        print(f"Evaluating {name} policy...")
        result = evaluate_policy(
            policy,
            factory,
            n_episodes=cfg.eval.n_episodes,
            base_seed=cfg.eval.base_seed,
            deterministic=cfg.eval.deterministic,
            metrics_cfg=cfg.metrics,
        )
        s = result["summary"]
        rows.append({
            "policy": name,
            "reward_mean": s.get("episode_reward_mean"),
            "reward_std": s.get("episode_reward_std"),
            "burned_cells_mean": s.get("burned_cells_mean"),
            "burned_cells_std": s.get("burned_cells_std"),
            "fire_intensity_mean": s.get("fire_intensity_mean"),
            "fire_intensity_std": s.get("fire_intensity_std"),
        })

        # Track trajectory steps over 1 sample episode for plotting
        sample_env = factory()
        obs, _ = sample_env.reset(seed=cfg.eval.base_seed)
        done = False
        fire_sizes = []
        agent_visits = np.zeros((grid_size, grid_size))
        
        while not done:
            action, _ = policy.predict(obs, deterministic=cfg.eval.deterministic)
            obs, _, terminated, truncated, _ = sample_env.step(action)
            fire_sizes.append(float(sample_env.state[0].sum()))
            
            # Record agent location
            ax, ay = sample_env.agent_pos
            agent_visits[ax, ay] += 1
            done = bool(terminated or truncated)

        trajectories[name] = fire_sizes
        influence_maps[name] = agent_visits

    metrics_df = pd.DataFrame(rows)
    metrics_out = ensure_dir(results_dir()) / "controllability_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print(f"Wrote controllability metrics -> {metrics_out}")

    # Generate Figures
    figs_dir = Path("figures")
    figs_dir.mkdir(exist_ok=True)

    # Figure 1: Fire Propagation Trajectories
    plt.figure(figsize=(8, 5))
    for name, path in trajectories.items():
        plt.plot(path, label=name, linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Fire Intensity (Sum)")
    plt.title("Fire Propagation Trajectories")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    traj_out = figs_dir / "fire_propagation_trajectories.png"
    plt.savefig(traj_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote trajectory plot -> {traj_out}")

    # Figure 2: Suppression Influence Maps (Nearest Fire vs Random vs NoOp)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, name in enumerate(["noop", "random", "nearest_fire"]):
        visits = influence_maps.get(name, np.zeros((grid_size, grid_size)))
        im = axes[idx].imshow(visits, cmap="hot", interpolation="nearest")
        axes[idx].set_title(f"Suppression Path: {name}")
        fig.colorbar(im, ax=axes[idx])
    influence_out = figs_dir / "suppression_influence_maps.png"
    plt.savefig(influence_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote suppression influence map -> {influence_out}")

    # Figure 3: Action Impact Visualizations (Delta from NoOp)
    plt.figure(figsize=(8, 5))
    noop_traj = np.array(trajectories["noop"])
    for name, path in trajectories.items():
        if name == "noop":
            continue
        min_len = min(len(noop_traj), len(path))
        delta = noop_traj[:min_len] - np.array(path[:min_len])
        plt.plot(delta, label=f"Delta NoOp - {name}", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Suppression Impact (Reduced Fire Cells)")
    plt.title("Action Suppression Impact Over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    impact_out = figs_dir / "action_impact_visualizations.png"
    plt.savefig(impact_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote action impact plot -> {impact_out}")

    # Figure 4: Controllability Bar Plots
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["policy"], metrics_df["fire_intensity_mean"], yerr=metrics_df["fire_intensity_std"], capsize=5, color="skyblue")
    plt.xlabel("Policy")
    plt.ylabel("Mean Fire Intensity (Lower is Better)")
    plt.title("Policy Performance Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    control_out = figs_dir / "controllability_plots.png"
    plt.savefig(control_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote controllability comparison bar chart -> {control_out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment/ablation.yaml")
    ap.add_argument("--set", nargs="*", help="OmegaConf overrides")
    args = ap.parse_args()
    
    run_controllability(args.config, args.set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
