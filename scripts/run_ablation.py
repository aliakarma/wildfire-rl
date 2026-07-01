#!/usr/bin/env python
"""Ablation study: toggle dynamics components and measure final fire.

Reproduces the intent of ``07a_experiments_and_ablations`` with one canonical env and a
single metric definition. Variants are defined in configs/experiment/ablation.yaml.

AAAI-grade: runs ALL seeds per variant, reports mean ± std, 95% CIs, and
significance tests (Welch's t-test) comparing each variant to baseline.

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
from wildfire_rl.eval.significance import confidence_interval_95, format_significance, welch_ttest
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
    baseline_fires: list[float] = []

    for variant, overrides in VARIANTS.items():
        env_cfg = dataclasses.replace(cfg.env, **overrides)
        factory = make_env_factory(state_tensor=tensor, config=env_cfg)

        seed_fires: list[float] = []
        seed_rewards: list[float] = []
        seed_burned: list[float] = []

        for seed in cfg.seeds:
            logger.info("Ablation '%s', seed=%d (overrides=%s)", variant, seed, overrides)
            model = train_ppo(factory, cfg.ppo, seed=seed)
            res = evaluate_policy(
                model,
                factory,
                n_episodes=cfg.eval.n_episodes,
                base_seed=cfg.eval.base_seed,
                metrics_cfg=cfg.metrics,
            )
            seed_fires.append(res["summary"]["fire_intensity_mean"])
            seed_rewards.append(res["summary"]["episode_reward_mean"])
            seed_burned.append(res["summary"]["burned_cells_mean"])

        fire_arr = np.array(seed_fires)
        reward_arr = np.array(seed_rewards)
        burned_arr = np.array(seed_burned)
        ci_fire = confidence_interval_95(fire_arr)
        ci_reward = confidence_interval_95(reward_arr)

        row = {
            "experiment": variant,
            "fire_mean": float(fire_arr.mean()),
            "fire_std": float(fire_arr.std()),
            "fire_ci_lo": ci_fire[0],
            "fire_ci_hi": ci_fire[1],
            "reward_mean": float(reward_arr.mean()),
            "reward_std": float(reward_arr.std()),
            "reward_ci_lo": ci_reward[0],
            "reward_ci_hi": ci_reward[1],
            "burned_mean": float(burned_arr.mean()),
            "burned_std": float(burned_arr.std()),
            "n_seeds": len(cfg.seeds),
        }

        if variant == "baseline":
            baseline_fires = seed_fires
        elif baseline_fires:
            test = welch_ttest(np.array(seed_fires), np.array(baseline_fires))
            row["p_vs_baseline"] = test["p_value"]
            row["d_vs_baseline"] = test["cohens_d"]
            row["sig_vs_baseline"] = format_significance(test["p_value"])
            logger.info(
                "  %s vs baseline: p=%.4f, d=%.2f (%s)",
                variant, test["p_value"], test["cohens_d"],
                format_significance(test["p_value"]),
            )

        rows.append(row)

    out = ensure_dir(results_dir()) / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
