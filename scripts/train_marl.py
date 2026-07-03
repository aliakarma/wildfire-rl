#!/usr/bin/env python
"""MARL cooperative scaling: train + evaluate across team sizes with a MATCHED budget.

Reproduces ``08_multi_agent_training`` / ``09_marl_scaling_experiments`` with one
canonical MultiAgentWildfireEnv and equal training budgets across agent counts (the
original used 20k for MARL vs 100k single-agent, which was not comparable).

AAAI-grade: runs ALL seeds per agent count, reports mean ± std, 95% CIs, and
significance tests (Welch's t-test) comparing each agent count to 1-agent baseline.

NOTE: 100k timesteps for 5 agents with MultiDiscrete(5^5 = 3125 possible actions) covers
the action space less densely than 100k for 1 agent with Discrete(5). This is a documented
confound — not corrected, to preserve fair wall-clock budget comparison. See Phase 5 notes.

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
from wildfire_rl.eval.significance import confidence_interval_95, format_significance, welch_ttest
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
    single_agent_rewards: list[float] = []

    for n_agents in cfg.marl.agent_counts:
        seed_rewards: list[float] = []
        seed_fires: list[float] = []
        seed_burned: list[float] = []

        for seed in cfg.seeds:
            factory = partial(_factory, tensor, cfg.env, n_agents)
            logger.info(
                "MARL: %d agents, seed=%d (budget=%d)", n_agents, seed, cfg.ppo.total_timesteps
            )
            save_path = models_dir() / f"ppo_marl_{cfg.region.name}_{n_agents}agents_seed_{seed}"
            model = train_ppo(factory, cfg.ppo, seed=seed, save_path=save_path)
            res = evaluate_policy(
                model,
                factory,
                n_episodes=cfg.eval.n_episodes,
                base_seed=cfg.eval.base_seed,
                metrics_cfg=cfg.metrics,
            )
            seed_rewards.append(res["summary"]["episode_reward_mean"])
            seed_fires.append(res["summary"]["fire_intensity_mean"])
            seed_burned.append(res["summary"]["burned_cells_mean"])

        reward_arr = np.array(seed_rewards)
        fire_arr = np.array(seed_fires)
        burned_arr = np.array(seed_burned)
        ci_reward = confidence_interval_95(reward_arr)
        ci_fire = confidence_interval_95(fire_arr)

        row = {
            "num_agents": n_agents,
            "reward_mean": float(reward_arr.mean()),
            "reward_std": float(reward_arr.std()),
            "reward_ci_lo": ci_reward[0],
            "reward_ci_hi": ci_reward[1],
            "fire_mean": float(fire_arr.mean()),
            "fire_std": float(fire_arr.std()),
            "fire_ci_lo": ci_fire[0],
            "fire_ci_hi": ci_fire[1],
            "burned_mean": float(burned_arr.mean()),
            "burned_std": float(burned_arr.std()),
            "n_seeds": len(cfg.seeds),
        }

        if n_agents == cfg.marl.agent_counts[0]:
            # First agent count is the baseline (typically 1)
            single_agent_rewards = seed_rewards
        elif single_agent_rewards:
            test = welch_ttest(np.array(seed_rewards), np.array(single_agent_rewards))
            row["p_vs_1agent"] = test["p_value"]
            row["d_vs_1agent"] = test["cohens_d"]
            row["sig_vs_1agent"] = format_significance(test["p_value"])
            logger.info(
                "  %d agents vs %d agent: p=%.4f, d=%.2f (%s)",
                n_agents,
                cfg.marl.agent_counts[0],
                test["p_value"],
                test["cohens_d"],
                format_significance(test["p_value"]),
            )

        rows.append(row)

    out = ensure_dir(results_dir()) / "marl_scaling_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
