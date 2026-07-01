#!/usr/bin/env python
"""Phase 2 evaluation driver: MARL Controllability Recovery & Cross-Regional Transfer.

Evaluates team sizes [1, 3, 5, 10] across NoOp, Random, NearestFire, Frontier, and PPO
under matched training budgets on Saudi and California. Performs significance testing,
coordination analysis, cross-regional transfer evaluation, and saves all tables and figures.
"""

from __future__ import annotations

import os
import sys
import json
import concurrent.futures
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add src/ to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from wildfire_rl.config import load_config
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv
from wildfire_rl.eval.baselines import NoOpPolicy, RandomPolicy, NearestFirePolicy, FrontierPolicy
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.loading import load_ppo_model
from wildfire_rl.eval.significance import welch_ttest, confidence_interval_95, format_significance, cohens_d
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir
from wildfire_rl.logging_utils import get_logger

logger = get_logger("run_marl_evaluation")

STAY_ACTION = 4


def train_model_worker(region_name: str, n_agents: int, seed: int, total_timesteps: int) -> str:
    """Worker process to train a PPO MARL model on CPU."""
    try:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
        import numpy as np
        from wildfire_rl.config import load_config
        from wildfire_rl.paths import region_tensor_path, models_dir
        from wildfire_rl.train.ppo import train_ppo
        from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv

        cfg = load_config()
        r_dir = "saudi_eastern_province" if region_name == "saudi" else "california"
        tensor = np.load(region_tensor_path(r_dir, 32))
        
        env_cfg = cfg.env
        env_cfg.reward_mode = "normalized"
        
        ppo_cfg = cfg.ppo
        ppo_cfg.total_timesteps = total_timesteps
        ppo_cfg.n_envs = 1  # DummyVecEnv
        
        def _factory():
            return MultiAgentWildfireEnv(state_tensor=tensor, config=env_cfg, num_agents=n_agents)
            
        save_path = models_dir() / f"ppo_marl_{region_name}_{n_agents}agents_seed_{seed}"
        train_ppo(_factory, ppo_cfg, seed=seed, save_path=save_path)
        return f"SUCCESS: {region_name}, {n_agents} agents, seed={seed}"
    except Exception as e:
        import traceback
        return f"FAILED: {region_name}, {n_agents} agents, seed={seed} - Error: {str(e)}\n{traceback.format_exc()}"


def train_missing_models(total_timesteps: int = 100000) -> None:
    """Detect and parallel train missing PPO MARL models."""
    regions = ["saudi", "california"]
    agent_counts = [1, 3, 5, 10]
    seeds = [0, 1, 2]
    
    jobs = []
    for reg in regions:
        for n_agents in agent_counts:
            for seed in seeds:
                model_file = models_dir() / f"ppo_marl_{reg}_{n_agents}agents_seed_{seed}.zip"
                if not model_file.exists():
                    jobs.append((reg, n_agents, seed, total_timesteps))
                    
    if not jobs:
        logger.info("All PPO MARL models are already trained.")
        return
        
    logger.info(f"Detected {len(jobs)} missing PPO MARL models. Launching parallel CPU training...")
    # Parallel CPU training using max 3 workers to keep CPU usage reasonable
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(train_model_worker, *job) for job in jobs]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            logger.info(res)


def run_evaluation() -> None:
    """Run Phase 2 evaluation pipeline."""
    cfg = load_config()
    regions = [
        {"name": "saudi", "dir": "saudi_eastern_province"},
        {"name": "california", "dir": "california"},
    ]
    agent_counts = [1, 3, 5, 10]
    seeds = [0, 1, 2]
    
    # Storage for all evaluation statistics
    raw_evals = [] # stores per-episode raw results
    
    # Special tracking for Saudi visual metrics
    saudi_ppo_visits = {}
    saudi_ppo_trajectories = {}
    saudi_ppo_entropies = {}
    saudi_ppo_repeated = {}
    saudi_ppo_agent_trajectories = {}

    for reg in regions:
        region_name = reg["name"]
        logger.info(f"\nEvaluating region: {region_name}")
        tensor = np.load(region_tensor_path(reg["dir"], 32))
        
        for n_agents in agent_counts:
            logger.info(f"  Team size: {n_agents} agents")
            
            # Setup environment factory
            env_cfg = cfg.env
            env_cfg.reward_mode = "normalized"
            
            def make_marl_factory(num_a):
                return lambda: MultiAgentWildfireEnv(state_tensor=tensor, config=env_cfg, num_agents=num_a)
                
            factory = make_marl_factory(n_agents)
            sample_env = factory()
            
            # Setup policies
            policies = {
                "noop": NoOpPolicy(sample_env.action_space),
                "random": RandomPolicy(sample_env.action_space, seed=cfg.seed),
                "nearest_fire": NearestFirePolicy(sample_env.action_space, grid_size=32, env=sample_env),
                "frontier": FrontierPolicy(sample_env.action_space, grid_size=32, env=sample_env),
            }
            
            # Load PPO models for the seeds
            ppo_models = []
            for seed in seeds:
                model_path = models_dir() / f"ppo_marl_{region_name}_{n_agents}agents_seed_{seed}.zip"
                if model_path.exists():
                    try:
                        ppo_models.append((seed, load_ppo_model(model_path, sample_env)))
                    except Exception as e:
                        logger.error(f"Failed to load PPO model at {model_path}: {e}")
                else:
                    logger.warning(f"Missing PPO model: {model_path}")
            
            # Evaluate baseline policies (NoOp, Random, NearestFire, Frontier)
            for pol_name, policy in policies.items():
                res = evaluate_policy(
                    policy,
                    factory,
                    n_episodes=cfg.eval.n_episodes,
                    base_seed=cfg.eval.base_seed,
                    deterministic=True,
                    metrics_cfg=cfg.metrics,
                )
                
                # Retrieve episode-level stats
                for ep_idx, ep in enumerate(res["episodes"]):
                    raw_evals.append({
                        "region": region_name,
                        "num_agents": n_agents,
                        "policy": pol_name,
                        "seed": 0,
                        "episode": ep_idx,
                        "reward": ep["episode_reward"],
                        "burned_cells": ep["burned_cells"],
                        "fire_intensity": ep["fire_intensity"],
                        "containment_efficiency": 1.0 - (ep["fire_intensity"] / (sample_env._initial_fire_total or 1.0)),
                    })
                    
            # Evaluate PPO across seeds
            for seed, model in ppo_models:
                res = evaluate_policy(
                    model,
                    factory,
                    n_episodes=cfg.eval.n_episodes,
                    base_seed=cfg.eval.base_seed,
                    deterministic=True,
                    metrics_cfg=cfg.metrics,
                )
                
                for ep_idx, ep in enumerate(res["episodes"]):
                    raw_evals.append({
                        "region": region_name,
                        "num_agents": n_agents,
                        "policy": "ppo",
                        "seed": seed,
                        "episode": ep_idx,
                        "reward": ep["episode_reward"],
                        "burned_cells": ep["burned_cells"],
                        "fire_intensity": ep["fire_intensity"],
                        "containment_efficiency": 1.0 - (ep["fire_intensity"] / (sample_env._initial_fire_total or 1.0)),
                    })

            # Run 1 detailed trajectory per seed for PPO/heuristics to measure coordination and specialization
            # For visualizing PPO on Saudi:
            for seed, model in ppo_models:
                run_detailed_diagnostics(
                    region_name, n_agents, seed, model, factory, 
                    saudi_ppo_visits, saudi_ppo_trajectories, 
                    saudi_ppo_entropies, saudi_ppo_repeated, 
                    saudi_ppo_agent_trajectories
                )

    df_raw = pd.DataFrame(raw_evals)
    
    # Save baseline comparisons & statistics
    save_baseline_comparisons(df_raw)
    
    # Save coordination & specialization statistics
    save_coordination_metrics(saudi_ppo_visits, saudi_ppo_repeated, saudi_ppo_entropies)

    # Run and save cross-regional transfer learning evaluation
    save_transfer_metrics(cfg, seeds)

    # Generate all plots
    generate_figures(df_raw, saudi_ppo_visits, saudi_ppo_trajectories, 
                     saudi_ppo_entropies, saudi_ppo_repeated, 
                     saudi_ppo_agent_trajectories)

    logger.info("\nMARL evaluation and reporting complete.")


def run_detailed_diagnostics(
    region_name: str, n_agents: int, seed: int, model: Any, env_factory: Any,
    saudi_ppo_visits: dict, saudi_ppo_trajectories: dict, 
    saudi_ppo_entropies: dict, saudi_ppo_repeated: dict,
    saudi_ppo_agent_trajectories: dict
) -> None:
    """Run one detailed episode to capture spatial trajectories, action entropy, and overlap."""
    env = env_factory()
    if hasattr(model, "env"):
        model.env = env
        
    obs, _ = env.reset(seed=seed)
    done = False
    
    visit_maps = [np.zeros((32, 32)) for _ in range(n_agents)]
    fire_sizes = [float(env.state[0].sum())]
    actions_taken = []
    agent_paths = [[] for _ in range(n_agents)]
    
    # Track starting positions
    for i in range(n_agents):
        agent_paths[i].append(tuple(env.agent_positions[i]))
        visit_maps[i][env.agent_positions[i][0], env.agent_positions[i][1]] += 1
        
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_arr = np.atleast_1d(action)
        actions_taken.append(action_arr)
        
        obs, _, terminated, truncated, _ = env.step(action)
        fire_sizes.append(float(env.state[0].sum()))
        
        for i in range(n_agents):
            px, py = env.agent_positions[i]
            agent_paths[i].append((px, py))
            visit_maps[i][px, py] += 1
            
        done = bool(terminated or truncated)
        
    actions_taken = np.array(actions_taken) # (steps, num_agents)
    n_steps = len(actions_taken)
    
    # Calculate action entropy & repeated action ratio per agent
    agent_entropies = []
    agent_repeated = []
    for i in range(n_agents):
        act_col = actions_taken[:, i]
        _, counts = np.unique(act_col, return_counts=True)
        probs = counts / n_steps
        ent = -np.sum(probs * np.log2(probs)) if n_steps > 0 else 0.0
        rep = np.sum(act_col[:-1] == act_col[1:]) / (n_steps - 1) if n_steps > 1 else 0.0
        agent_entropies.append(ent)
        agent_repeated.append(rep)
        
    # Aggregate Saudi logs for visualization
    if region_name == "saudi":
        key = (n_agents, seed)
        saudi_ppo_visits[key] = visit_maps
        saudi_ppo_trajectories[key] = fire_sizes
        saudi_ppo_entropies[key] = np.mean(agent_entropies)
        saudi_ppo_repeated[key] = np.mean(agent_repeated)
        saudi_ppo_agent_trajectories[key] = agent_paths


def save_baseline_comparisons(df_raw: pd.DataFrame) -> None:
    """Perform significance testing and save baseline statistics & effect sizes."""
    # Table 1: Baseline Statistics (marl_baseline_statistics.csv)
    stats_rows = []
    
    # Table 2: Effect Sizes (marl_effect_sizes.csv)
    effect_rows = []
    
    # Table 3: Controllability Metrics (marl_controllability_metrics.csv)
    controllability_rows = []
    
    for (region, n_agents), group in df_raw.groupby(["region", "num_agents"]):
        # Group evaluations by policy
        ppo_reward = group[group["policy"] == "ppo"]["reward"].values
        ppo_burned = group[group["policy"] == "ppo"]["burned_cells"].values
        ppo_fire = group[group["policy"] == "ppo"]["fire_intensity"].values
        
        # Calculate PPO stats
        ppo_ci_lo, ppo_ci_hi = confidence_interval_95(ppo_reward)
        stats_rows.append({
            "region": region,
            "num_agents": n_agents,
            "policy": "ppo",
            "reward_mean": ppo_reward.mean(),
            "reward_std": ppo_reward.std(),
            "reward_ci_lo": ppo_ci_lo,
            "reward_ci_hi": ppo_ci_hi,
            "burned_cells_mean": ppo_burned.mean(),
            "fire_intensity_mean": ppo_fire.mean(),
        })
        
        # Heuristics & baselines
        for pol in ["noop", "random", "nearest_fire", "frontier"]:
            pol_group = group[group["policy"] == pol]
            pol_reward = pol_group["reward"].values
            pol_burned = pol_group["burned_cells"].values
            pol_fire = pol_group["fire_intensity"].values
            
            ci_lo, ci_hi = confidence_interval_95(pol_reward)
            stats_rows.append({
                "region": region,
                "num_agents": n_agents,
                "policy": pol,
                "reward_mean": pol_reward.mean(),
                "reward_std": pol_reward.std(),
                "reward_ci_lo": ci_lo,
                "reward_ci_hi": ci_hi,
                "burned_cells_mean": pol_burned.mean(),
                "fire_intensity_mean": pol_fire.mean(),
            })
            
            # Welch's t-test comparing PPO to baseline
            if len(ppo_reward) > 1 and len(pol_reward) > 1:
                t_test = welch_ttest(ppo_reward, pol_reward)
                d = cohens_d(ppo_reward, pol_reward)
                effect_rows.append({
                    "region": region,
                    "num_agents": n_agents,
                    "comparison": f"ppo_vs_{pol}",
                    "t_statistic": t_test["t_statistic"],
                    "p_value": t_test["p_value"],
                    "cohens_d": d,
                    "significant": bool(t_test["p_value"] < 0.05),
                })
                
        # Controllability metrics
        # Suppression coverage and containment efficiency
        # We estimate these from PPO
        ppo_cont_eff = group[group["policy"] == "ppo"]["containment_efficiency"].values
        controllability_rows.append({
            "region": region,
            "num_agents": n_agents,
            "containment_efficiency_mean": ppo_cont_eff.mean(),
            "containment_efficiency_std": ppo_cont_eff.std(),
            "burned_cells_mean": ppo_burned.mean(),
            "fire_intensity_mean": ppo_fire.mean(),
        })

    # Save to CSV
    pd.DataFrame(stats_rows).to_csv(ensure_dir(results_dir()) / "marl_baseline_statistics.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(results_dir() / "marl_effect_sizes.csv", index=False)
    pd.DataFrame(controllability_rows).to_csv(results_dir() / "marl_controllability_metrics.csv", index=False)
    
    logger.info("Wrote baseline stats, effect sizes, and controllability metrics under results/")


def save_coordination_metrics(saudi_ppo_visits: dict, saudi_ppo_repeated: dict, saudi_ppo_entropies: dict) -> None:
    """Compute and save coordination, overlap, and specialization metrics."""
    rows = []
    for n_agents in [1, 3, 5, 10]:
        seeds = [0, 1, 2]
        overlap_ratios = []
        coverage_efficiencies = []
        specializations = []
        
        for s in seeds:
            key = (n_agents, s)
            if key not in saudi_ppo_visits:
                continue
                
            visit_maps = saudi_ppo_visits[key]
            
            # 1. Overlap ratio: spatial correlation of agent visit heatmaps
            # For n_agents=1, overlap is trivially 0.0
            if n_agents == 1:
                overlap = 0.0
                spec = 1.0
                eff = np.count_nonzero(visit_maps[0]) / (visit_maps[0].sum() or 1.0)
            else:
                # Average pairwise overlap: intersection of visited coordinates
                visited_sets = [set(zip(*np.nonzero(vm))) for vm in visit_maps]
                pairwise_overlaps = []
                for i in range(n_agents):
                    for j in range(i + 1, n_agents):
                        intersect = len(visited_sets[i].intersection(visited_sets[j]))
                        union = len(visited_sets[i].union(visited_sets[j]))
                        pairwise_overlaps.append(intersect / union if union > 0 else 0.0)
                overlap = np.mean(pairwise_overlaps)
                
                # Region specialization: 1.0 - mean spatial correlation between visit heatmaps
                pairwise_corrs = []
                for i in range(n_agents):
                    for j in range(i + 1, n_agents):
                        flat_i = visit_maps[i].flatten()
                        flat_j = visit_maps[j].flatten()
                        if flat_i.sum() > 0 and flat_j.sum() > 0:
                            corr = np.corrcoef(flat_i, flat_j)[0, 1]
                            if np.isnan(corr):
                                corr = 0.0
                        else:
                            corr = 0.0
                        pairwise_corrs.append(corr)
                spec = 1.0 - max(0.0, np.mean(pairwise_corrs))
                
                # Coverage efficiency: union of visited cells / sum of steps taken by all agents
                union_visited = set()
                for vs in visited_sets:
                    union_visited.update(vs)
                total_steps = sum(vm.sum() for vm in visit_maps)
                eff = len(union_visited) / (total_steps or 1.0)
                
            overlap_ratios.append(overlap)
            specializations.append(spec)
            coverage_efficiencies.append(eff)
            
        if overlap_ratios:
            # Average across seeds
            rows.append({
                "num_agents": n_agents,
                "overlap_ratio_mean": np.mean(overlap_ratios),
                "overlap_ratio_std": np.std(overlap_ratios),
                "coverage_efficiency_mean": np.mean(coverage_efficiencies),
                "region_specialization_mean": np.mean(specializations),
                "action_entropy_mean": np.mean([saudi_ppo_entropies[(n_agents, s)] for s in seeds if (n_agents, s) in saudi_ppo_entropies]),
                "repeated_action_ratio_mean": np.mean([saudi_ppo_repeated[(n_agents, s)] for s in seeds if (n_agents, s) in saudi_ppo_repeated]),
            })

    pd.DataFrame(rows).to_csv(ensure_dir(results_dir()) / "coordination_metrics.csv", index=False)
    logger.info("Wrote coordination metrics -> results/coordination_metrics.csv")


def save_transfer_metrics(cfg: Any, seeds: list[int]) -> None:
    """Evaluate cross-regional transfer matrices using 5-agent models."""
    logger.info("\nEvaluating cross-regional transfer for 5-agent PPO models...")
    
    # We load Saudi model and California model (n_agents=5), and evaluate on both environments.
    env_cfg = cfg.env
    env_cfg.reward_mode = "normalized"
    
    saudi_tensor = np.load(region_tensor_path("saudi_eastern_province", 32))
    california_tensor = np.load(region_tensor_path("california", 32))
    
    saudi_factory = lambda: MultiAgentWildfireEnv(state_tensor=saudi_tensor, config=env_cfg, num_agents=5)
    california_factory = lambda: MultiAgentWildfireEnv(state_tensor=california_tensor, config=env_cfg, num_agents=5)
    
    transfer_results = []
    
    # Matrix of (train_region, test_region)
    pairs = [
        ("saudi", "saudi", saudi_factory),
        ("saudi", "california", california_factory),
        ("california", "california", california_factory),
        ("california", "saudi", saudi_factory),
    ]
    
    for train_reg, test_reg, factory in pairs:
        logger.info(f"  Transfer: Train={train_reg} -> Test={test_reg}")
        
        rewards = []
        burned = []
        for seed in seeds:
            model_path = models_dir() / f"ppo_marl_{train_reg}_5agents_seed_{seed}.zip"
            if model_path.exists():
                try:
                    sample_env = factory()
                    model = load_ppo_model(model_path, sample_env)
                    res = evaluate_policy(
                        model,
                        factory,
                        n_episodes=cfg.eval.n_episodes,
                        base_seed=cfg.eval.base_seed,
                        deterministic=True,
                        metrics_cfg=cfg.metrics,
                    )
                    rewards.extend([ep["episode_reward"] for ep in res["episodes"]])
                    burned.extend([ep["burned_cells"] for ep in res["episodes"]])
                except Exception as e:
                    logger.error(f"Failed to evaluate transfer for {train_reg} on {test_reg}: {e}")
                    
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
    
    # Compute transfer degradation %
    # Degradation = (Native_Reward - Transfer_Reward) / abs(Native_Reward)
    saudi_native = df_trans[(df_trans["train_region"] == "saudi") & (df_trans["test_region"] == "saudi")]["reward_mean"].values[0]
    saudi_transfer = df_trans[(df_trans["train_region"] == "california") & (df_trans["test_region"] == "saudi")]["reward_mean"].values[0]
    
    ca_native = df_trans[(df_trans["train_region"] == "california") & (df_trans["test_region"] == "california")]["reward_mean"].values[0]
    ca_transfer = df_trans[(df_trans["train_region"] == "saudi") & (df_trans["test_region"] == "california")]["reward_mean"].values[0]
    
    saudi_degradation = (saudi_native - saudi_transfer) / abs(saudi_native) * 100
    ca_degradation = (ca_native - ca_transfer) / abs(ca_native) * 100
    
    logger.info(f"Saudi Transfer Degradation: {saudi_degradation:.2f}%")
    logger.info(f"California Transfer Degradation: {ca_degradation:.2f}%")
    
    df_trans["transfer_degradation_pct"] = [
        0.0 if train == test else (saudi_degradation if test == "saudi" else ca_degradation)
        for train, test in zip(df_trans["train_region"], df_trans["test_region"])
    ]
    
    df_trans.to_csv(ensure_dir(results_dir()) / "marl_transfer_statistics.csv", index=False)
    logger.info("Wrote transfer statistics -> results/marl_transfer_statistics.csv")


def generate_figures(
    df_raw: pd.DataFrame, saudi_ppo_visits: dict, saudi_ppo_trajectories: dict,
    saudi_ppo_entropies: dict, saudi_ppo_repeated: dict,
    saudi_ppo_agent_trajectories: dict
) -> None:
    """Generate and save publication-quality figures."""
    figs_dir = Path("figures")
    figs_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. figures/marl_controllability_scaling.png
    plt.figure(figsize=(8, 5))
    for pol in ["noop", "random", "nearest_fire", "frontier", "ppo"]:
        pol_data = df_raw[(df_raw["region"] == "saudi") & (df_raw["policy"] == pol)]
        if pol_data.empty:
            continue
        means = []
        ci_los = []
        ci_his = []
        agents = [1, 3, 5, 10]
        for n in agents:
            sub = pol_data[pol_data["num_agents"] == n]["fire_intensity"].values
            if len(sub) > 0:
                means.append(sub.mean())
                lo, hi = confidence_interval_95(sub)
                ci_los.append(lo)
                ci_his.append(hi)
            else:
                means.append(np.nan)
                ci_los.append(np.nan)
                ci_his.append(np.nan)
        
        plt.plot(agents, means, marker="o", label=pol.upper(), linewidth=2)
        plt.fill_between(agents, ci_los, ci_his, alpha=0.15)
        
    plt.xlabel("Team Size (Number of Agents)", fontsize=12)
    plt.ylabel("Fire Intensity (Final Sum)", fontsize=12)
    plt.title("Saudi Arabia Wildfire Controllability Scaling Curves", fontsize=13, fontweight="bold")
    plt.xticks([1, 3, 5, 10])
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(figs_dir / "marl_controllability_scaling.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated: figures/marl_controllability_scaling.png")

    # 2. figures/spatial_coverage_maps.png
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    sizes = [1, 3, 5, 10]
    for idx, n in enumerate(sizes):
        agg_map = np.zeros((32, 32))
        count = 0
        for s in [0, 1, 2]:
            key = (n, s)
            if key in saudi_ppo_visits:
                for vm in saudi_ppo_visits[key]:
                    agg_map += vm
                count += 1
        if count > 0:
            agg_map /= count
            im = axes[idx].imshow(agg_map, cmap="rocket_r", interpolation="nearest")
            axes[idx].set_title(f"PPO: {n} Agents", fontsize=14, fontweight="bold")
            axes[idx].grid(False)
            fig.colorbar(im, ax=axes[idx])
        else:
            axes[idx].axis("off")
            
    fig.suptitle("Spatial Suppression Visit Density Heatmaps (Saudi PPO)", fontsize=16, fontweight="bold", y=1.05)
    plt.savefig(figs_dir / "spatial_coverage_maps.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated: figures/spatial_coverage_maps.png")

    # 3. figures/marl_entropy_analysis.png
    plt.figure(figsize=(7, 4.5))
    agents = [1, 3, 5, 10]
    entropies = []
    repeated = []
    for n in agents:
        ents = [saudi_ppo_entropies[(n, s)] for s in [0, 1, 2] if (n, s) in saudi_ppo_entropies]
        reps = [saudi_ppo_repeated[(n, s)] for s in [0, 1, 2] if (n, s) in saudi_ppo_repeated]
        entropies.append(np.mean(ents) if ents else 0.0)
        repeated.append(np.mean(reps) if reps else 0.0)
        
    x = np.arange(len(agents))
    width = 0.35
    plt.bar(x - width/2, entropies, width, label='Action Entropy (Bits)', color='royalblue')
    plt.bar(x + width/2, repeated, width, label='Repeated Action Ratio', color='tomato')
    plt.ylabel('Value', fontsize=12)
    plt.xlabel('Team Size (Number of Agents)', fontsize=12)
    plt.title('PPO Policy Collapse Diagnostics vs Team Size (Saudi)', fontsize=13, fontweight="bold")
    plt.xticks(x, [str(a) for a in agents])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(figs_dir / "marl_entropy_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated: figures/marl_entropy_analysis.png")

    # 4. figures/marl_trajectory_diversity.png
    plt.figure(figsize=(8, 5))
    if (1, 0) in saudi_ppo_trajectories:
        plt.plot(saudi_ppo_trajectories[(1, 0)], label="PPO: 1 Agent", color="gray", linewidth=2)
    if (5, 0) in saudi_ppo_trajectories:
        plt.plot(saudi_ppo_trajectories[(5, 0)], label="PPO: 5 Agents (Seed 0)", color="blue", linewidth=2.5)
    if (5, 1) in saudi_ppo_trajectories:
        plt.plot(saudi_ppo_trajectories[(5, 1)], label="PPO: 5 Agents (Seed 1)", color="cyan", linewidth=2.5)
    if (5, 2) in saudi_ppo_trajectories:
        plt.plot(saudi_ppo_trajectories[(5, 2)], label="PPO: 5 Agents (Seed 2)", color="purple", linewidth=2.5)
        
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Fire Intensity (Sum)", fontsize=12)
    plt.title("Wildfire Propagation Trajectories (Saudi Arabia)", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(figs_dir / "marl_trajectory_diversity.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated: figures/marl_trajectory_diversity.png")

    # 5. figures/coordination_heatmaps.png
    key = (5, 0)
    if key in saudi_ppo_visits:
        visit_maps = saudi_ppo_visits[key]
        corr_matrix = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                flat_i = visit_maps[i].flatten()
                flat_j = visit_maps[j].flatten()
                if flat_i.sum() > 0 and flat_j.sum() > 0:
                    corr = np.corrcoef(flat_i, flat_j)[0, 1]
                    corr_matrix[i, j] = 0.0 if np.isnan(corr) else corr
                else:
                    corr_matrix[i, j] = 0.0
                    
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1.0, vmax=1.0, fmt=".2f")
        plt.title("Saudi PPO: Inter-Agent Spatial Visit Correlation Matrix (N=5)", fontsize=12, fontweight="bold")
        plt.xlabel("Agent ID")
        plt.ylabel("Agent ID")
        plt.savefig(figs_dir / "coordination_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/coordination_heatmaps.png")

    # 6. figures/marl_transfer_heatmap.png
    trans_stat_path = results_dir() / "marl_transfer_statistics.csv"
    if trans_stat_path.exists():
        df_trans = pd.read_csv(trans_stat_path)
        pivot = df_trans.pivot(index="train_region", columns="test_region", values="reward_mean")
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, cmap="YlOrRd_r", fmt=".2f", cbar_kws={'label': 'Normalized Reward'})
        plt.title("Cross-Regional Transfer Matrix Heatmap (5 Agents)", fontsize=13, fontweight="bold")
        plt.xlabel("Test Region")
        plt.ylabel("Train Region")
        plt.savefig(figs_dir / "marl_transfer_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/marl_transfer_heatmap.png")

    # 7. figures/marl_transfer_degradation.png
    if trans_stat_path.exists():
        df_trans = pd.read_csv(trans_stat_path)
        df_deg = df_trans[df_trans["train_region"] != df_trans["test_region"]]
        
        plt.figure(figsize=(6, 4.5))
        bars = plt.bar(df_deg["train_region"] + " -> " + df_deg["test_region"], 
                       df_deg["transfer_degradation_pct"], color=["orange", "darkred"])
        plt.ylabel("Reward Degradation (%)", fontsize=12)
        plt.title("Cross-Regional Transfer Degradation % (5 Agents)", fontsize=13, fontweight="bold")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 1.0, f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            
        plt.ylim(0, max(df_deg["transfer_degradation_pct"]) + 10)
        plt.savefig(figs_dir / "marl_transfer_degradation.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/marl_transfer_degradation.png")

    # 8. figures/agent_specialization_maps.png
    key3 = (3, 0)
    if key3 in saudi_ppo_agent_trajectories:
        paths = saudi_ppo_agent_trajectories[key3]
        plt.figure(figsize=(6, 6))
        saudi_tensor = np.load(region_tensor_path("saudi_eastern_province", 32))
        plt.imshow(saudi_tensor[0], cmap="gray_r", alpha=0.3, extent=[0, 32, 32, 0])
        
        colors = ["blue", "green", "red"]
        for idx, path in enumerate(paths):
            xs, ys = zip(*path)
            plt.plot(ys, xs, label=f"Agent {idx}", color=colors[idx], linewidth=1.5, marker="o", markersize=3, alpha=0.8)
            plt.scatter(ys[0], xs[0], color=colors[idx], marker="s", s=60, edgecolors='black', zorder=5)
            
        plt.title("Agent Spatial Path Trajectories: Division of Labor (N=3)", fontsize=12, fontweight="bold")
        plt.xlabel("Grid Column")
        plt.ylabel("Grid Row")
        plt.legend()
        plt.xlim(0, 32)
        plt.ylim(32, 0)
        plt.savefig(figs_dir / "agent_specialization_maps.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/agent_specialization_maps.png")

    # 9. figures/coordination_efficiency.png
    coord_path = results_dir() / "coordination_metrics.csv"
    if coord_path.exists():
        df_coord = pd.read_csv(coord_path)
        
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        
        ax1.plot(df_coord["num_agents"], df_coord["coverage_efficiency_mean"], marker="s", color="darkgreen", linewidth=2.5, label="Unique Coverage per Step")
        ax1.set_xlabel("Team Size (Number of Agents)", fontsize=12)
        ax1.set_ylabel("Coverage Efficiency (Unique cells / steps)", color="darkgreen", fontsize=12)
        ax1.tick_params(axis='y', labelcolor="darkgreen")
        ax1.set_xticks([1, 3, 5, 10])
        
        ax2 = ax1.twinx()
        ax2.plot(df_coord["num_agents"], df_coord["region_specialization_mean"], marker="^", color="darkblue", linewidth=2.5, label="Region Specialization")
        ax2.set_ylabel("Region Specialization Metric", color="darkblue", fontsize=12)
        ax2.tick_params(axis='y', labelcolor="darkblue")
        
        plt.title("Coordination Efficiency & Spatial Specialization vs Team Size", fontsize=12, fontweight="bold")
        fig.tight_layout()
        plt.savefig(figs_dir / "coordination_efficiency.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Generated: figures/coordination_efficiency.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100000, help="Timesteps to train missing PPO models")
    args = ap.parse_args()
    
    # 1. Train any missing models (budget-matched)
    train_missing_models(total_timesteps=args.timesteps)
    
    # 2. Run the full MARL scientific evaluation pipeline
    run_evaluation()
