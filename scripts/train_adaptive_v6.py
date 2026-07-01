#!/usr/bin/env python
"""Train a single Adaptive Hybrid model from the command line.

Uses the Adaptive coordination environment with dynamic wildfire scenarios.
"""

import argparse
import numpy as np
from wildfire_rl.config import load_config
from wildfire_rl.coordination.adaptive_coordination import AdaptiveHybridMultiAgentWildfireEnv
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir
from wildfire_rl.train.ppo import train_ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=["saudi", "california"])
    ap.add_argument("--agents", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--config", type=str, default="configs/experiment/phase6_dynamic.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    r_dir = "saudi_eastern_province" if args.region == "saudi" else "california"
    tensor = np.load(region_tensor_path(r_dir, 32))
    ensure_dir(models_dir())

    env_cfg = cfg.env
    env_cfg.reward_mode = "normalized"
    env_cfg.reward_v3.enabled = True

    # PPO hyperparameters matching requirements
    ppo_cfg = cfg.ppo
    ppo_cfg.total_timesteps = args.timesteps
    ppo_cfg.n_envs = 1

    def _factory():
        return AdaptiveHybridMultiAgentWildfireEnv(
            state_tensor=tensor,
            config=env_cfg,
            num_agents=args.agents,
        )

    # TensorBoard logging
    tb_log = str(ensure_dir(results_dir() / "runs" / "v6"))

    save_path = models_dir() / f"ppo_adaptive_marl_{args.region}_{args.agents}agents_seed_{args.seed}"
    train_ppo(_factory, ppo_cfg, seed=args.seed, save_path=save_path, tensorboard_log=tb_log)
    print(f"FINISHED: {args.region}, {args.agents} agents, seed={args.seed}")


if __name__ == "__main__":
    main()
