#!/usr/bin/env python
"""Ablation study: toggle dynamics components and measure final fire.

Reproduces the intent of ``07a_experiments_and_ablations`` with one canonical env and a
single metric definition. Variants are defined in configs/experiment/ablation.yaml.

    python scripts/run_ablation.py --config configs/experiment/ablation.yaml
    python scripts/run_ablation.py --config configs/experiment/ablation.yaml --set ppo.total_timesteps=2000
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from wildfire_rl.config import load_config
from wildfire_rl.envs.base import make_env_factory
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.paths import ensure_dir, region_tensor_path, results_dir
from wildfire_rl.train.ppo import train_ppo

logger = get_logger("run_ablation")

# variant -> EnvConfig field overrides
VARIANTS = {
    "baseline": {},
    "no_wind": {"wind_coeff": 0.0},
    "no_terrain": {"terrain_coeff": 0.0},
    "no_suppression": {"suppression_factor": 1.0},
    "dense_fuel": {"fuel_coeff": 0.30},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment/ablation.yaml")
    ap.add_argument("--set", dest="overrides", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    tensor = np.load(region_tensor_path(cfg.region.dir, cfg.region.grid_size))

    rows = []
    for variant, overrides in VARIANTS.items():
        env_cfg = dataclasses.replace(cfg.env, **overrides)
        factory = make_env_factory(state_tensor=tensor, config=env_cfg)
        seed = cfg.seeds[0]
        logger.info("Ablation '%s' (overrides=%s)", variant, overrides)
        model = train_ppo(factory, cfg.ppo, seed=seed)
        res = evaluate_policy(
            model, factory, n_episodes=cfg.eval.n_episodes,
            base_seed=cfg.eval.base_seed, metrics_cfg=cfg.metrics,
        )
        rows.append({
            "experiment": variant,
            "final_mean_fire": res["summary"]["fire_intensity_mean"],
            "mean_burned_cells": res["summary"]["burned_cells_mean"],
        })

    out = ensure_dir(results_dir()) / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
