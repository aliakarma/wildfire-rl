"""Orchestrate the full cross-region transfer matrix from a config + trained models.

Locates one trained model per region (``models/ppo_<region>_<grid>_seed_<seed>.zip``),
builds an evaluation env factory per region, and computes the complete matrix via
:func:`wildfire_rl.eval.transfer.transfer_matrix`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wildfire_rl.config import load_config, to_dict
from wildfire_rl.envs.base import make_env_factory
from wildfire_rl.eval.baselines import RandomPolicy
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
            from stable_baselines3 import PPO

            logger.info("Loaded %s policy <- %s", region.name, model_path)
            policies[region.name] = PPO.load(str(model_path))

    df = transfer_matrix(
        policies, env_factories,
        n_episodes=cfg.eval.n_episodes, base_seed=cfg.eval.base_seed,
        deterministic=cfg.eval.deterministic, metrics_cfg=cfg.metrics,
    )

    out = ensure_dir(results_dir()) / "transfer_matrix.csv"
    df.to_csv(out, index=False)
    write_run_metadata(results_dir() / "runs" / "transfer.json", config_dict=to_dict(cfg), seed=seed)
    logger.info("Wrote transfer matrix -> %s\n%s", out, df.to_string(index=False))
    return out
