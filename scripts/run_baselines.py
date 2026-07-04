"""Benchmark script to evaluate all baseline policies on Saudi and California landscapes (Phase 4).

Runs heuristics, no-op, random, and trained Maskable-PPO agents. Computes statistical
significance (paired t-test and Welch's t-test) and bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from wildfire_marl.env.single_agent_env import FireSuppressionEnv
from wildfire_marl.env.rewards import InfrastructureWeightedReward, FireSizeReward
from wildfire_marl.eval.evaluate import evaluate_policy
from wildfire_marl.eval.significance import (
    bootstrap_ci,
    welch_ttest,
    format_ci,
    format_mean_std,
    format_significance,
)
from wildfire_marl.agents.heuristics import (
    NoOpPolicy,
    RandomPolicy,
    NearestFrontierPolicy,
    GreatestRiskFirstPolicy,
    ValueWeightedFirstPolicy,
)
from wildfire_marl.agents.ppo_baseline import train_ppo, wrap_env_for_maskable_ppo, SB3PredictorWrapper


def run_benchmark(
    regions: list[str],
    episodes: int = 20,
    train_timesteps: int = 10000,
    base_seed: int = 42,
    output_dir: str | Path = "results/runs",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all summary rows for output CSV
    all_summary_rows = []

    for region in regions:
        print("\n" + "=" * 60)
        print(f"BENCHMARKING REGION: {region.upper()}")
        print("=" * 60)
        
        map_name = "Saudi" if region == "saudi" else "California"
        
        # Factory for environment setup
        def env_factory():
            return FireSuppressionEnv(
                fire_map=map_name,
                data_dir="data/cell2fire",
                steps_per_action=60,  # hourly control
                reward_cls=InfrastructureWeightedReward,
                observe_infra=True,
                catastrophe_weight=2.0,
                cascade_prob=0.1,  # include cascade probability
            )

        # 1. Define Policies
        policies = {
            "No-Op": NoOpPolicy(),
            "Random": RandomPolicy(),
            "Frontier Heuristic": NearestFrontierPolicy(),
            "Greatest Risk": GreatestRiskFirstPolicy(),
            "Value-Weighted Frontier": ValueWeightedFirstPolicy(),
        }

        # 2. Train and include PPO Policy
        ppo_model_path = output_dir / f"ppo_{region}_seed_{base_seed}.zip"
        if train_timesteps > 0:
            print(f"Training Maskable-PPO baseline agent for {train_timesteps} steps...")
            ppo_model = train_ppo(
                region=region,
                total_timesteps=train_timesteps,
                seed=base_seed,
                save_path=ppo_model_path,
                observe_infra=True,
                catastrophe_weight=2.0,
                cascade_prob=0.1,
            )
            policies["Maskable-PPO"] = SB3PredictorWrapper(ppo_model)
        elif ppo_model_path.exists():
            print(f"Loading existing Maskable-PPO model from {ppo_model_path}...")
            from sb3_contrib import MaskablePPO
            ppo_model = MaskablePPO.load(ppo_model_path)
            policies["Maskable-PPO"] = SB3PredictorWrapper(ppo_model)

        # 3. Evaluate each policy
        eval_results = {}
        for name, policy in policies.items():
            print(f"\nEvaluating policy: {name} over {episodes} episodes...")
            res = evaluate_policy(
                policy=policy,
                env_factory=env_factory,
                n_episodes=episodes,
                base_seed=base_seed,
                scenario_seed_offset=100000,  # disjoint ignition seeds
            )
            eval_results[name] = res
            
            summary = res["summary"]
            print(f"  Reward: {summary['episode_reward_mean']:.2f} ± {summary['episode_reward_std']:.2f}")
            print(f"  Burned Cells: {summary['burned_cells_mean']:.2f} ± {summary['burned_cells_std']:.2f}")
            if "wel_mean" in summary:
                print(f"  Economic Loss (WEL): {summary['wel_mean']:.2f} ± {summary['wel_std']:.2f}")
                print(f"  Survival Rate (ISR): {summary['isr_mean']:.2f} ± {summary['isr_std']:.2f}")

        # 4. Statistical significance tests against No-Op baseline
        no_op_rewards = np.array([ep["episode_reward"] for ep in eval_results["No-Op"]["episodes"]])
        
        for name, res in eval_results.items():
            ep_rewards = np.array([ep["episode_reward"] for ep in res["episodes"]])
            ep_burned = np.array([ep["burned_cells"] for ep in res["episodes"]])
            ep_wel = np.array([ep.get("wel", 0.0) for ep in res["episodes"]])
            
            # Welch's t-test vs No-Op (rewards)
            t_test = welch_ttest(ep_rewards, no_op_rewards) if name != "No-Op" else {
                "t_statistic": 0.0, "p_value": 1.0, "cohens_d": 0.0
            }
            
            # Bootstrap CIs of the mean
            reward_ci = bootstrap_ci(ep_rewards, seed=base_seed)
            burned_ci = bootstrap_ci(ep_burned, seed=base_seed)
            wel_ci = bootstrap_ci(ep_wel, seed=base_seed) if "wel" in res["episodes"][0] else (0.0, 0.0)

            # Store summary row
            all_summary_rows.append({
                "region": region,
                "policy": name,
                "reward_mean": ep_rewards.mean(),
                "reward_std": ep_rewards.std(),
                "reward_ci_lo": reward_ci[0],
                "reward_ci_hi": reward_ci[1],
                "burned_mean": ep_burned.mean(),
                "burned_std": ep_burned.std(),
                "burned_ci_lo": burned_ci[0],
                "burned_ci_hi": burned_ci[1],
                "wel_mean": ep_wel.mean() if "wel" in res["episodes"][0] else 0.0,
                "wel_ci_lo": wel_ci[0],
                "wel_ci_hi": wel_ci[1],
                "t_stat_vs_noop": t_test["t_statistic"],
                "p_val_vs_noop": t_test["p_value"],
                "cohens_d_vs_noop": t_test["cohens_d"],
                "sig_stars": format_significance(t_test["p_value"]) if name != "No-Op" else "n/a",
            })

    # Save summary as CSV
    summary_df = pd.DataFrame(all_summary_rows)
    csv_path = output_dir / "baselines_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nAll baseline results saved successfully to {csv_path}")

    # Generate Markdown Table Report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT (Mean [95% CI])")
    print("=" * 60)
    for region in regions:
        print(f"\nRegion: {region.upper()}")
        reg_df = summary_df[summary_df["region"] == region]
        for _, row in reg_df.iterrows():
            print(
                f"  {row['policy']:<25}: "
                f"Reward {row['reward_mean']:7.2f} [{row['reward_ci_lo']:.2f}, {row['reward_ci_hi']:.2f}] | "
                f"Burned {row['burned_mean']:6.1f} [{row['burned_ci_lo']:.1f}, {row['burned_ci_hi']:.1f}] | "
                f"Effect Size d={row['cohens_d_vs_noop']:5.2f} {row['sig_stars']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Baseline Benchmarking Script.")
    parser.add_argument("--region", default="both", choices=["saudi", "california", "both"], help="Region.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--train-timesteps", type=int, default=10000, help="Timesteps to train Maskable-PPO.")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation random seed base.")
    args = parser.parse_args()

    regions = ["saudi", "california"] if args.region == "both" else [args.region]
    
    run_benchmark(
        regions=regions,
        episodes=args.episodes,
        train_timesteps=args.train_timesteps,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
