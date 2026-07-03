#!/usr/bin/env python
"""Phase 2.5 evaluation driver: PPO-V2 Recovery & Stabilization.

Evaluates V2 PPO models against baselines and V1 PPO, generates diagnostic
plots for entropy, action diversity, reward components, coordination, and
produces publication-quality comparison figures.
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

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from wildfire_rl.config import load_config
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv
from wildfire_rl.envs.multi_agent_v2 import MultiAgentWildfireEnvV2
from wildfire_rl.eval.baselines import (
    NoOpPolicy, RandomPolicy, NearestFirePolicy, FrontierPolicy,
)
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.loading import load_ppo_model
from wildfire_rl.eval.significance import welch_ttest, confidence_interval_95, cohens_d
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir
from wildfire_rl.logging_utils import get_logger

logger = get_logger("run_marl_v2_evaluation")


def run_detailed_episode(
    model: Any, env_factory: Any, seed: int = 0
) -> dict[str, Any]:
    """Run one episode with full diagnostic tracking."""
    env = env_factory()
    if hasattr(model, "env"):
        model.env = env

    obs, _ = env.reset(seed=seed)
    done = False

    n_agents = env.num_agents
    visit_maps = [np.zeros((32, 32)) for _ in range(n_agents)]
    actions_taken = []
    fire_sizes = [float(env.state[0].sum())]
    reward_components_log = []
    agent_paths = [[] for _ in range(n_agents)]

    for i in range(n_agents):
        agent_paths[i].append(tuple(env.agent_positions[i]))
        visit_maps[i][env.agent_positions[i][0], env.agent_positions[i][1]] += 1

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_arr = np.atleast_1d(action)
        actions_taken.append(action_arr.copy())

        obs, _, terminated, truncated, info = env.step(action)
        fire_sizes.append(float(env.state[0].sum()))

        if "reward_components" in info:
            reward_components_log.append(dict(info["reward_components"]))

        for i in range(n_agents):
            px, py = env.agent_positions[i]
            agent_paths[i].append((px, py))
            visit_maps[i][px, py] += 1

        done = bool(terminated or truncated)

    actions_arr = np.array(actions_taken)
    n_steps = len(actions_arr)

    # Per-agent entropy and repeated-action ratio
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

    # Action distribution
    action_counts = np.zeros(5)
    for a in actions_arr.flatten():
        action_counts[int(a)] += 1
    action_dist = action_counts / max(action_counts.sum(), 1)

    return {
        "visit_maps": visit_maps,
        "fire_sizes": fire_sizes,
        "actions_taken": actions_arr,
        "agent_entropies": agent_entropies,
        "agent_repeated": agent_repeated,
        "mean_entropy": float(np.mean(agent_entropies)),
        "mean_repeated": float(np.mean(agent_repeated)),
        "action_distribution": action_dist,
        "reward_components_log": reward_components_log,
        "agent_paths": agent_paths,
    }


def run_evaluation() -> None:
    """Full V2 evaluation pipeline."""
    cfg = load_config("configs/ppo/marl_v2.yaml")
    regions = [
        {"name": "saudi", "dir": "saudi_eastern_province"},
        {"name": "california", "dir": "california"},
    ]
    agent_counts = [3, 5, 10]
    seeds = [0, 1, 2]
    ent_coefs = [0.005, 0.01, 0.02, 0.05]

    v2_results_dir = ensure_dir(results_dir() / "v2")
    v2_figs_dir = ensure_dir(Path("figures") / "v2")
    sns.set_theme(style="whitegrid")

    # ── 1. Entropy Sweep Analysis ──
    logger.info("=== Entropy Coefficient Sweep Analysis ===")
    sweep_rows = []
    sweep_diagnostics = {}

    for ent_coef in ent_coefs:
        for seed in seeds:
            model_name = f"ppo_v2_marl_saudi_5agents_ent{ent_coef}_seed_{seed}.zip"
            model_path = models_dir() / model_name
            if not model_path.exists():
                logger.warning(f"Missing: {model_name}")
                continue

            tensor = np.load(region_tensor_path("saudi_eastern_province", 32))
            env_cfg = cfg.env
            env_cfg.reward_mode = "normalized"
            env_cfg.reward_v2.enabled = True

            factory = lambda t=tensor, c=env_cfg: MultiAgentWildfireEnvV2(
                state_tensor=t, config=c, num_agents=5
            )
            sample_env = factory()

            try:
                model = load_ppo_model(model_path, sample_env)
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                continue

            diag = run_detailed_episode(model, factory, seed=seed)
            sweep_diagnostics[(ent_coef, seed)] = diag

            # Evaluate on 20 episodes
            res = evaluate_policy(
                model, factory, n_episodes=20,
                base_seed=0, deterministic=True, metrics_cfg=cfg.metrics,
            )

            for ep_idx, ep in enumerate(res["episodes"]):
                sweep_rows.append({
                    "ent_coef": ent_coef,
                    "seed": seed,
                    "episode": ep_idx,
                    "reward": ep["episode_reward"],
                    "burned_cells": ep["burned_cells"],
                    "fire_intensity": ep["fire_intensity"],
                    "mean_entropy": diag["mean_entropy"],
                    "mean_repeated": diag["mean_repeated"],
                })

    if sweep_rows:
        df_sweep = pd.DataFrame(sweep_rows)
        df_sweep.to_csv(v2_results_dir / "entropy_sweep_results.csv", index=False)
        logger.info("Saved entropy sweep results")

        # Plot: Entropy vs ent_coef
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1a. Entropy curve
        ent_means = []
        ent_stds = []
        for ec in ent_coefs:
            vals = [sweep_diagnostics[(ec, s)]["mean_entropy"]
                    for s in seeds if (ec, s) in sweep_diagnostics]
            ent_means.append(np.mean(vals) if vals else 0)
            ent_stds.append(np.std(vals) if vals else 0)

        axes[0].errorbar(ent_coefs, ent_means, yerr=ent_stds, marker="o",
                        capsize=5, linewidth=2, color="royalblue")
        axes[0].set_xlabel("Entropy Coefficient", fontsize=12)
        axes[0].set_ylabel("Action Entropy (Bits)", fontsize=12)
        axes[0].set_title("Policy Entropy vs ent_coef", fontsize=13, fontweight="bold")
        axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Collapse")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # 1b. Repeated-action ratio
        rep_means = []
        for ec in ent_coefs:
            vals = [sweep_diagnostics[(ec, s)]["mean_repeated"]
                    for s in seeds if (ec, s) in sweep_diagnostics]
            rep_means.append(np.mean(vals) if vals else 1.0)

        axes[1].bar([str(e) for e in ent_coefs], rep_means, color="tomato", alpha=0.85)
        axes[1].set_xlabel("Entropy Coefficient", fontsize=12)
        axes[1].set_ylabel("Repeated Action Ratio", fontsize=12)
        axes[1].set_title("Action Repetition vs ent_coef", fontsize=13, fontweight="bold")
        axes[1].axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Full Collapse")
        axes[1].legend()
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(axis="y", linestyle="--", alpha=0.6)

        # 1c. Action distribution heatmap
        dist_data = []
        for ec in ent_coefs:
            dists = [sweep_diagnostics[(ec, s)]["action_distribution"]
                     for s in seeds if (ec, s) in sweep_diagnostics]
            if dists:
                dist_data.append(np.mean(dists, axis=0))
            else:
                dist_data.append(np.zeros(5))

        action_labels = ["Up", "Down", "Left", "Right", "Stay"]
        dist_df = pd.DataFrame(dist_data, index=[str(e) for e in ent_coefs],
                              columns=action_labels)
        sns.heatmap(dist_df, annot=True, cmap="YlOrRd", fmt=".2f", ax=axes[2],
                   vmin=0, vmax=1, cbar_kws={"label": "Action Probability"})
        axes[2].set_xlabel("Action", fontsize=12)
        axes[2].set_ylabel("Entropy Coefficient", fontsize=12)
        axes[2].set_title("Action Diversity Heatmap", fontsize=13, fontweight="bold")

        plt.suptitle("PPO-V2 Entropy Sweep (Saudi, 5 Agents)", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(v2_figs_dir / "entropy_sweep_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/v2/entropy_sweep_analysis.png")

    # ── 2. Full V2 Evaluation (best ent_coef) ──
    logger.info("\n=== Full V2 Evaluation ===")
    raw_evals = []
    v2_diagnostics = {}

    for reg in regions:
        region_name = reg["name"]
        tensor = np.load(region_tensor_path(reg["dir"], 32))
        logger.info(f"\nEvaluating region: {region_name}")

        for n_agents in agent_counts:
            logger.info(f"  Team size: {n_agents} agents")

            env_cfg = cfg.env
            env_cfg.reward_mode = "normalized"
            env_cfg.reward_v2.enabled = True

            # V2 factory
            def make_v2_factory(t, c, na):
                return lambda: MultiAgentWildfireEnvV2(
                    state_tensor=t, config=c, num_agents=na
                )

            # V1 factory (for comparison)
            def make_v1_factory(t, c, na):
                return lambda: MultiAgentWildfireEnv(
                    state_tensor=t, config=c, num_agents=na
                )

            v2_factory = make_v2_factory(tensor, env_cfg, n_agents)
            v1_factory = make_v1_factory(tensor, env_cfg, n_agents)
            sample_env = v2_factory()

            # Baseline policies (evaluated on V2 env)
            policies = {
                "noop": NoOpPolicy(sample_env.action_space),
                "random": RandomPolicy(sample_env.action_space, seed=cfg.seed),
                "nearest_fire": NearestFirePolicy(sample_env.action_space, grid_size=32),
                "frontier": FrontierPolicy(sample_env.action_space, grid_size=32),
            }

            for pol_name, policy in policies.items():
                res = evaluate_policy(
                    policy, v2_factory, n_episodes=20,
                    base_seed=0, deterministic=True, metrics_cfg=cfg.metrics,
                )
                for ep_idx, ep in enumerate(res["episodes"]):
                    raw_evals.append({
                        "region": region_name, "num_agents": n_agents,
                        "policy": pol_name, "version": "baseline",
                        "seed": 0, "episode": ep_idx,
                        "reward": ep["episode_reward"],
                        "burned_cells": ep["burned_cells"],
                        "fire_intensity": ep["fire_intensity"],
                    })

            # V1 PPO models
            for seed in seeds:
                v1_path = models_dir() / f"ppo_marl_{region_name}_{n_agents}agents_seed_{seed}.zip"
                if v1_path.exists():
                    try:
                        v1_env = v1_factory()
                        v1_model = load_ppo_model(v1_path, v1_env)
                        res = evaluate_policy(
                            v1_model, v1_factory, n_episodes=20,
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

                        # Detailed diagnostics for V1
                        diag = run_detailed_episode(v1_model, v1_factory, seed=seed)
                        v2_diagnostics[("v1", region_name, n_agents, seed)] = diag
                    except Exception as e:
                        logger.error(f"Failed to load V1 model {v1_path}: {e}")

            # V2 PPO models (try all ent_coefs, use best available)
            for ent_coef in ent_coefs:
                for seed in seeds:
                    v2_path = models_dir() / f"ppo_v2_marl_{region_name}_{n_agents}agents_ent{ent_coef}_seed_{seed}.zip"
                    if v2_path.exists():
                        try:
                            v2_env = v2_factory()
                            v2_model = load_ppo_model(v2_path, v2_env)
                            res = evaluate_policy(
                                v2_model, v2_factory, n_episodes=20,
                                base_seed=0, deterministic=True, metrics_cfg=cfg.metrics,
                            )
                            for ep_idx, ep in enumerate(res["episodes"]):
                                raw_evals.append({
                                    "region": region_name, "num_agents": n_agents,
                                    "policy": f"ppo_v2_ent{ent_coef}", "version": "v2",
                                    "seed": seed, "episode": ep_idx,
                                    "reward": ep["episode_reward"],
                                    "burned_cells": ep["burned_cells"],
                                    "fire_intensity": ep["fire_intensity"],
                                })

                            diag = run_detailed_episode(v2_model, v2_factory, seed=seed)
                            v2_diagnostics[("v2", region_name, n_agents, ent_coef, seed)] = diag
                        except Exception as e:
                            logger.error(f"Failed to load V2 model {v2_path}: {e}")

    if raw_evals:
        df_raw = pd.DataFrame(raw_evals)
        df_raw.to_csv(v2_results_dir / "v2_evaluation_results.csv", index=False)
        logger.info("Saved V2 evaluation results")

        # Generate comparison figures
        generate_comparison_figures(df_raw, v2_diagnostics, v2_figs_dir)

    logger.info("\nV2 evaluation complete.")


def generate_comparison_figures(
    df_raw: pd.DataFrame,
    diagnostics: dict,
    figs_dir: Path,
) -> None:
    """Generate publication-quality comparison figures."""
    sns.set_theme(style="whitegrid")

    # ── Figure 1: V1 vs V2 Reward Comparison ──
    saudi_data = df_raw[df_raw["region"] == "saudi"]
    if not saudi_data.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Reward comparison
        policies_to_plot = [p for p in saudi_data["policy"].unique()
                           if p in ["noop", "nearest_fire", "ppo_v1"] or p.startswith("ppo_v2")]
        for pol in policies_to_plot:
            pol_data = saudi_data[saudi_data["policy"] == pol]
            means = []
            ci_los, ci_his = [], []
            agents = sorted(pol_data["num_agents"].unique())
            for n in agents:
                vals = pol_data[pol_data["num_agents"] == n]["fire_intensity"].values
                if len(vals) > 0:
                    means.append(vals.mean())
                    lo, hi = confidence_interval_95(vals)
                    ci_los.append(lo)
                    ci_his.append(hi)

            label = pol.upper().replace("_", " ")
            axes[0].plot(agents, means, marker="o", label=label, linewidth=2)
            axes[0].fill_between(agents, ci_los, ci_his, alpha=0.15)

        axes[0].set_xlabel("Team Size", fontsize=12)
        axes[0].set_ylabel("Fire Intensity (Final)", fontsize=12)
        axes[0].set_title("Saudi: Fire Intensity Scaling", fontsize=13, fontweight="bold")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # Entropy comparison V1 vs V2
        v1_entropies = {}
        v2_entropies = {}
        for key, diag in diagnostics.items():
            if key[0] == "v1" and key[1] == "saudi":
                n_agents = key[2]
                v1_entropies.setdefault(n_agents, []).append(diag["mean_entropy"])
            elif key[0] == "v2" and key[1] == "saudi":
                n_agents = key[2]
                v2_entropies.setdefault(n_agents, []).append(diag["mean_entropy"])

        agents_list = sorted(set(list(v1_entropies.keys()) + list(v2_entropies.keys())))
        if agents_list:
            x = np.arange(len(agents_list))
            width = 0.35
            v1_means = [np.mean(v1_entropies.get(a, [0])) for a in agents_list]
            v2_means = [np.mean(v2_entropies.get(a, [0])) for a in agents_list]

            axes[1].bar(x - width/2, v1_means, width, label="PPO V1", color="gray", alpha=0.7)
            axes[1].bar(x + width/2, v2_means, width, label="PPO V2", color="royalblue", alpha=0.85)
            axes[1].set_xlabel("Team Size", fontsize=12)
            axes[1].set_ylabel("Action Entropy (Bits)", fontsize=12)
            axes[1].set_title("Entropy: V1 vs V2", fontsize=13, fontweight="bold")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([str(a) for a in agents_list])
            axes[1].legend()
            axes[1].grid(axis="y", linestyle="--", alpha=0.6)

        plt.suptitle("PPO V1 vs V2 Comparison (Saudi)", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(figs_dir / "v1_vs_v2_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/v2/v1_vs_v2_comparison.png")

    # ── Figure 2: Reward Component Contributions ──
    component_data = {}
    for key, diag in diagnostics.items():
        if key[0] == "v2" and "reward_components_log" in diag and diag["reward_components_log"]:
            n_agents = key[2] if len(key) > 2 else 5
            comps = diag["reward_components_log"]
            avg_comps = {}
            for comp_name in comps[0].keys():
                if comp_name not in ("total_reward",):
                    avg_comps[comp_name] = np.mean([c.get(comp_name, 0) for c in comps])
            component_data.setdefault(n_agents, []).append(avg_comps)

    if component_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        agents_sorted = sorted(component_data.keys())
        comp_names = ["base_reward", "spread_reduction", "frontier_blocking",
                      "coverage", "overlap_penalty", "containment_stability"]
        colors = ["#636363", "#2171b5", "#e6550d", "#31a354", "#d62728", "#756bb1"]

        bar_data = {cn: [] for cn in comp_names}
        for n in agents_sorted:
            all_comps = component_data[n]
            for cn in comp_names:
                vals = [c.get(cn, 0) for c in all_comps]
                bar_data[cn].append(np.mean(vals))

        x = np.arange(len(agents_sorted))
        width = 0.12
        for i, (cn, color) in enumerate(zip(comp_names, colors)):
            ax.bar(x + i * width, bar_data[cn], width, label=cn.replace("_", " ").title(),
                  color=color, alpha=0.85)

        ax.set_xlabel("Team Size", fontsize=12)
        ax.set_ylabel("Mean Reward Component", fontsize=12)
        ax.set_title("V2 Reward Component Contributions", fontsize=13, fontweight="bold")
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels([str(a) for a in agents_sorted])
        ax.legend(fontsize=9, ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(figs_dir / "reward_component_contributions.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/v2/reward_component_contributions.png")

    # ── Figure 3: Overlap & Coverage Comparison ──
    v1_overlaps = {}
    v2_overlaps = {}
    v2_coverages = {}
    for key, diag in diagnostics.items():
        visit_maps = diag["visit_maps"]
        n_agents = key[2] if len(key) > 2 else 5

        if len(visit_maps) > 1:
            visited_sets = [set(zip(*np.nonzero(vm))) for vm in visit_maps]
            pairwise = []
            for i in range(len(visited_sets)):
                for j in range(i+1, len(visited_sets)):
                    inter = len(visited_sets[i].intersection(visited_sets[j]))
                    union = len(visited_sets[i].union(visited_sets[j]))
                    pairwise.append(inter / union if union > 0 else 0)
            overlap = np.mean(pairwise)

            union_all = set()
            for vs in visited_sets:
                union_all.update(vs)
            total_steps = sum(vm.sum() for vm in visit_maps)
            coverage_eff = len(union_all) / max(total_steps, 1)
        else:
            overlap = 0.0
            coverage_eff = np.count_nonzero(visit_maps[0]) / max(visit_maps[0].sum(), 1)

        if key[0] == "v1":
            v1_overlaps.setdefault(n_agents, []).append(overlap)
        elif key[0] == "v2":
            v2_overlaps.setdefault(n_agents, []).append(overlap)
            v2_coverages.setdefault(n_agents, []).append(coverage_eff)

    if v1_overlaps or v2_overlaps:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        agents_sorted = sorted(set(list(v1_overlaps.keys()) + list(v2_overlaps.keys())))

        if agents_sorted:
            x = np.arange(len(agents_sorted))
            width = 0.35
            v1_ov = [np.mean(v1_overlaps.get(a, [0])) for a in agents_sorted]
            v2_ov = [np.mean(v2_overlaps.get(a, [0])) for a in agents_sorted]

            axes[0].bar(x - width/2, v1_ov, width, label="PPO V1", color="gray", alpha=0.7)
            axes[0].bar(x + width/2, v2_ov, width, label="PPO V2", color="royalblue", alpha=0.85)
            axes[0].set_xlabel("Team Size", fontsize=12)
            axes[0].set_ylabel("Overlap Ratio (IoU)", fontsize=12)
            axes[0].set_title("Agent Spatial Overlap", fontsize=13, fontweight="bold")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([str(a) for a in agents_sorted])
            axes[0].legend()
            axes[0].grid(axis="y", linestyle="--", alpha=0.6)

            v2_cov = [np.mean(v2_coverages.get(a, [0])) for a in agents_sorted]
            axes[1].bar([str(a) for a in agents_sorted], v2_cov, color="darkgreen", alpha=0.85)
            axes[1].set_xlabel("Team Size", fontsize=12)
            axes[1].set_ylabel("Coverage Efficiency", fontsize=12)
            axes[1].set_title("V2 Coverage Efficiency", fontsize=13, fontweight="bold")
            axes[1].grid(axis="y", linestyle="--", alpha=0.6)

        plt.suptitle("Coordination Analysis: V1 vs V2", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(figs_dir / "overlap_coverage_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/v2/overlap_coverage_comparison.png")

    logger.info("\nAll V2 figures generated.")


if __name__ == "__main__":
    run_evaluation()
