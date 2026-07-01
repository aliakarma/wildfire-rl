#!/usr/bin/env python
"""Train a single MARL V2 model from the command line.

Uses the V2 environment with enhanced reward shaping and supports
entropy coefficient overrides for sweep experiments.
"""

import argparse
import numpy as np
from wildfire_rl.config import load_config
from wildfire_rl.envs.multi_agent_v2 import MultiAgentWildfireEnvV2
from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir
from wildfire_rl.train.ppo import train_ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=["saudi", "california"])
    ap.add_argument("--agents", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--ent-coef", type=float, default=None,
                    help="Override entropy coefficient (for sweep experiments)")
    ap.add_argument("--config", type=str, default="configs/ppo/marl_v2.yaml",
                    help="Path to V2 config YAML")
    args = ap.parse_args()

    cfg = load_config(args.config)
    r_dir = "saudi_eastern_province" if args.region == "saudi" else "california"
    tensor = np.load(region_tensor_path(r_dir, 32))
    ensure_dir(models_dir())

    env_cfg = cfg.env
    env_cfg.reward_mode = "normalized"
    # Ensure V2 reward is enabled
    env_cfg.reward_v2.enabled = True

    ppo_cfg = cfg.ppo
    ppo_cfg.total_timesteps = args.timesteps
    ppo_cfg.n_envs = 1

    # Override ent_coef if specified (for sweep experiments)
    if args.ent_coef is not None:
        ppo_cfg.ent_coef = args.ent_coef

    ent_str = f"_ent{ppo_cfg.ent_coef}"

    def _factory():
        return MultiAgentWildfireEnvV2(
            state_tensor=tensor, config=env_cfg, num_agents=args.agents
        )

    # TensorBoard logging
    tb_log = str(ensure_dir(results_dir() / "runs" / "v2"))

    save_path = models_dir() / f"ppo_v2_marl_{args.region}_{args.agents}agents{ent_str}_seed_{args.seed}"
    train_ppo(_factory, ppo_cfg, seed=args.seed, save_path=save_path, tensorboard_log=tb_log)
    print(f"FINISHED: {args.region}, {args.agents} agents, ent_coef={ppo_cfg.ent_coef}, seed={args.seed}")


if __name__ == "__main__":
    main()
