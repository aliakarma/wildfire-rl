#!/usr/bin/env python
"""MARL cooperative scaling: train + evaluate across team sizes with a MATCHED budget.

Reproduces ``08_multi_agent_training`` / ``09_marl_scaling_experiments`` with one
canonical MultiAgentWildfireEnv and equal training budgets across agent counts (the
original used 20k for MARL vs 100k single-agent, which was not comparable).

    python scripts/train_marl.py --config configs/experiment/scaling.yaml
"""

from __future__ import annotations

import argparse
from functools import partial

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from wildfire_rl.config import load_config
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir
from wildfire_rl.train.ppo import train_ppo

logger = get_logger("train_marl")


def _factory(tensor, env_cfg, num_agents):
    return MultiAgentWildfireEnv(state_tensor=tensor, config=env_cfg, num_agents=num_agents)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment/scaling.yaml")
    ap.add_argument("--set", dest="overrides", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    tensor = np.load(region_tensor_path(cfg.region.dir, cfg.region.grid_size))
    ensure_dir(models_dir())

    rows = []
    for n_agents in cfg.marl.agent_counts:
        factory = partial(_factory, tensor, cfg.env, n_agents)
        seed = cfg.seeds[0]
        logger.info("MARL scaling: %d agents (budget=%d)", n_agents, cfg.ppo.total_timesteps)
        save_path = models_dir() / f"ppo_marl_{cfg.region.name}_{n_agents}agents_seed_{seed}"
        model = train_ppo(factory, cfg.ppo, seed=seed, save_path=save_path)
        res = evaluate_policy(
            model, factory, n_episodes=cfg.eval.n_episodes,
            base_seed=cfg.eval.base_seed, metrics_cfg=cfg.metrics,
        )
        rows.append({
            "num_agents": n_agents,
            "final_mean_fire": res["summary"]["fire_intensity_mean"],
            "mean_reward": res["summary"]["episode_reward_mean"],
            "mean_burned_cells": res["summary"]["burned_cells_mean"],
        })

    out = ensure_dir(results_dir()) / "marl_scaling_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
