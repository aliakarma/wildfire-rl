"""Orchestrate the full cross-region transfer matrix from a config + trained models.

Locates one trained model per region (``models/ppo_<region>_<grid>_seed_<seed>.zip``),
builds an evaluation env factory per region, and computes the complete matrix via
:func:`wildfire_rl.eval.transfer.transfer_matrix`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_rl.config import load_config, to_dict
from wildfire_rl.envs.base import make_env_factory
from wildfire_rl.eval.baselines import RandomPolicy
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.transfer import transfer_matrix
from wildfire_rl.logging_utils import get_logger, write_run_metadata
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir

logger = get_logger("wildfire_rl.experiments.transfer")


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


def run_transfer(
    config_path: str | None = None,
    overrides: list[str] | None = None,
    seed: int = 0,
) -> Path:
    """Compute and save the transfer matrix CSV; returns the output path."""
    cfg = load_config(config_path, overrides)
    if not cfg.regions:
        raise ValueError("transfer requires `regions:` in the config (see configs/experiment/transfer.yaml).")

    policies = {}
    env_factories = {}
    
    from wildfire_rl.eval.baselines import RandomPolicy, NoOpPolicy
    from wildfire_rl.eval.significance import paired_ttest, format_significance

    for region in cfg.regions:
        tensor = np.load(region_tensor_path(region.dir, region.grid_size))
        env_factories[region.name] = make_env_factory(state_tensor=tensor, config=cfg.env)

        model_path = _find_model(region.name, region.grid_size, seed)
        if model_path is None:
            logger.warning(
                "No trained model for '%s' (looked in %s). Using RandomPolicy as a "
                "placeholder so the pipeline runs end-to-end; train models for real results.",
                region.name, models_dir(),
            )
            policies[region.name] = RandomPolicy(env_factories[region.name]().action_space, seed=seed)
        else:
            from wildfire_rl.eval.loading import load_ppo_model
            logger.info("Loaded %s policy <- %s", region.name, model_path)
            policies[region.name] = load_ppo_model(model_path, env_factories[region.name]())

        # Add baselines for this region
        policies[f"random_{region.name}"] = RandomPolicy(env_factories[region.name]().action_space, seed=seed)
        policies[f"noop_{region.name}"] = NoOpPolicy(env_factories[region.name]().action_space)

    rows = []
    episode_rewards = {}

    for train_name, policy in policies.items():
        for test_name, factory in env_factories.items():
            # If baseline is region-specific, only run on that region
            if "_" in train_name:
                base_type, base_region = train_name.split("_", 1)
                if base_region != test_name:
                    continue

            result = evaluate_policy(
                policy,
                factory,
                n_episodes=cfg.eval.n_episodes,
                base_seed=cfg.eval.base_seed,
                deterministic=cfg.eval.deterministic,
                metrics_cfg=cfg.metrics,
            )
            s = result["summary"]
            rewards = [ep["episode_reward"] for ep in result["episodes"]]
            episode_rewards[(train_name, test_name)] = rewards

            rows.append({
                "train_region": train_name,
                "test_region": test_name,
                "mean_reward": s.get("episode_reward_mean"),
                "reward_std": s.get("episode_reward_std"),
                "mean_burned_cells": s.get("burned_cells_mean"),
                "mean_fire_intensity": s.get("fire_intensity_mean"),
                "n_episodes": cfg.eval.n_episodes,
            })

    # Paired t-tests: compare native PPO vs transferred PPO
    for row in rows:
        train = row["train_region"]
        test = row["test_region"]
        if train in env_factories and train != test:
            native_key = (test, test)
            transfer_key = (train, test)
            if native_key in episode_rewards and transfer_key in episode_rewards:
                native_rewards = np.array(episode_rewards[native_key])
                transfer_rewards = np.array(episode_rewards[transfer_key])
                test_res = paired_ttest(transfer_rewards, native_rewards)
                row["p_vs_native"] = test_res["p_value"]
                row["d_vs_native"] = test_res["cohens_d"]
                row["sig_vs_native"] = format_significance(test_res["p_value"])
                logger.info(
                    "  Transfer %s -> %s vs Native: p=%.4f, d=%.2f (%s)",
                    train, test, test_res["p_value"], test_res["cohens_d"],
                    format_significance(test_res["p_value"])
                )

    df = pd.DataFrame(rows)
    out_raw = ensure_dir(results_dir()) / "transfer_matrix_raw.csv"
    out_norm = ensure_dir(results_dir()) / "transfer_matrix.csv"
    
    # Save both or appropriate name
    df.to_csv(out_norm if cfg.env.reward_mode == "normalized" else out_raw, index=False)
    write_run_metadata(results_dir() / "runs" / f"transfer_{cfg.env.reward_mode}.json", config_dict=to_dict(cfg), seed=seed)
    logger.info("Wrote transfer matrix -> %s\n%s", out_norm if cfg.env.reward_mode == "normalized" else out_raw, df.to_string(index=False))
    return out_norm if cfg.env.reward_mode == "normalized" else out_raw
