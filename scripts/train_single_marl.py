#!/usr/bin/env python
"""Train a single MARL model from the command line.

Allows robust parallelization using independent OS processes in Google Colab.
"""

import argparse
import numpy as np
from functools import partial
from wildfire_rl.config import load_config
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path
from wildfire_rl.train.ppo import train_ppo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=["saudi", "california"])
    ap.add_argument("--agents", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=100000)
    args = ap.parse_args()

    cfg = load_config()
    r_dir = "saudi_eastern_province" if args.region == "saudi" else "california"
    tensor = np.load(region_tensor_path(r_dir, 32))
    ensure_dir(models_dir())

    env_cfg = cfg.env
    env_cfg.reward_mode = "normalized"

    ppo_cfg = cfg.ppo
    ppo_cfg.total_timesteps = args.timesteps
    ppo_cfg.n_envs = 1
    ppo_cfg.ent_coef = 0.05  # Tuned exploration to prevent MARL collapse

    def _factory():
        return MultiAgentWildfireEnv(state_tensor=tensor, config=env_cfg, num_agents=args.agents)

    save_path = models_dir() / f"ppo_marl_{args.region}_{args.agents}agents_seed_{args.seed}"
    train_ppo(_factory, ppo_cfg, seed=args.seed, save_path=save_path)
    print(f"FINISHED: {args.region}, {args.agents} agents, seed={args.seed}")

if __name__ == "__main__":
    main()
