#!/usr/bin/env python
"""Phase 17 — per-ablation rollout visualization.

Renders one deterministic single-agent episode per (policy × environmental-ablation variant) to
``figures/rollouts/<variant>_<policy>.png``: the fire field at key timesteps, the agent trajectory,
and the Saudi criticality overlay. Makes behavioral differences legible — in particular the honest
PPO **collapse** (near-constant action, ≈ no-op) versus the heuristic routers that contain the fire.

Policies: ppo (authoritative Saudi checkpoint), nearest_fire, frontier, noop.
Variants:  baseline, no_wind, no_terrain, no_suppression, dense_fuel (env-dynamics ablations).

    python scripts/render_rollouts.py
    python scripts/render_rollouts.py --policies ppo frontier --variants baseline no_wind
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

import _bootstrap  # noqa: F401
from wildfire_rl.config import EnvConfig, InfraConfig
from wildfire_rl.envs.base import WildfireEnv
from wildfire_rl.eval.baselines import FrontierPolicy, NearestFirePolicy, NoOpPolicy
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.paths import ensure_dir, figures_dir, models_dir, region_tensor_path
from wildfire_rl.viz.rollout import record_episode, render_rollout

logger = get_logger("render_rollouts")
GRID = 32
REGION_DIR = "saudi_eastern_province"

# Environmental-dynamics ablation variants -> EnvConfig overrides.
VARIANTS = {
    "baseline": {},
    "no_wind": {"wind_coeff": 0.0},
    "no_terrain": {"terrain_coeff": 0.0},
    "no_suppression": {"suppression_factor": 1.0},  # suppression has no effect
    "dense_fuel": {"fuel_coeff": 0.6},  # much higher fuel-driven spread
}
POLICIES = ["ppo", "nearest_fire", "frontier", "noop"]


def _base_cfg(overrides: dict) -> EnvConfig:
    # Saudi desert dynamics; 8-channel obs (observe_infra off) so the authoritative PPO model loads.
    cfg = EnvConfig(
        spread_scale=0.7,
        ignition_rate=0.05,
        n_ignition_points=6,
        randomize_ignition=True,
        criticality_path=f"data/{REGION_DIR}/grids/32x32/criticality.npy",
    )
    cfg.infra = InfraConfig(infra_dir=f"data/{REGION_DIR}/grids/32x32/infrastructure")
    return replace(cfg, **overrides)


def _make_policy(name, env, model_cache):
    a, g = env.action_space, env.grid_size
    if name == "nearest_fire":
        return NearestFirePolicy(a, grid_size=g, env=env)
    if name == "frontier":
        return FrontierPolicy(a, grid_size=g, env=env)
    if name == "noop":
        return NoOpPolicy(a)
    if name == "ppo":
        if "ppo" not in model_cache:
            from wildfire_rl.eval.loading import load_ppo_model

            path = models_dir() / "ppo_saudi_32_seed_0.zip"
            if not path.exists():
                return None
            model_cache["ppo"] = load_ppo_model(path, env)
        return model_cache["ppo"]
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policies", nargs="*", default=POLICIES)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    tensor = np.load(region_tensor_path(REGION_DIR, GRID))
    out_dir = ensure_dir(figures_dir() / "rollouts")
    model_cache: dict = {}

    for variant in args.variants:
        cfg = _base_cfg(VARIANTS[variant])
        for policy_name in args.policies:
            env = WildfireEnv(state_tensor=tensor, config=cfg)
            policy = _make_policy(policy_name, env, model_cache)
            if policy is None:
                logger.warning("Skipping %s/%s (no PPO checkpoint)", variant, policy_name)
                continue
            rec = record_episode(env, policy, seed=args.seed, max_steps=args.steps)
            out = out_dir / f"{variant}_{policy_name}.png"
            render_rollout(rec, out, title=f"{policy_name} · {variant} · seed={args.seed}")
            logger.info("Wrote %s (%d frames)", out.name, len(rec["frames"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
