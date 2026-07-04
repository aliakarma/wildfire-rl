#!/usr/bin/env python
"""Render strategic rollouts per (policy family × region) — Phase 15B.5.

For each family (nearest_fire / frontier / hierarchical_greedy / hierarchical_risk_aware) and region,
writes a canonical animated GIF (``viz/rollout.py``) and a strategic filmstrip PNG (``viz/strategic.py``)
under ``figures/strategic/``, so infrastructure-defense behavior is visually legible.

    python scripts/render_strategic.py
    python scripts/render_strategic.py --regions saudi --families hierarchical_risk_aware --steps 40
"""

from __future__ import annotations

import argparse

import numpy as np
from omegaconf import OmegaConf

import _bootstrap  # noqa: F401
from wildfire_rl.config import EnvConfig, InfraConfig
from wildfire_rl.coordination.strategic_controller import StrategicController
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv
from wildfire_rl.eval.baselines import FrontierPolicy, NearestFirePolicy
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.paths import ensure_dir, figures_dir, region_tensor_path
from wildfire_rl.viz.rollout import record_rollout, render_gif
from wildfire_rl.viz.strategic import plot_strategic_filmstrip

logger = get_logger("render_strategic")
GRID = 32


def _env(rc, num_agents):
    tensor = np.load(region_tensor_path(rc.dir, GRID))
    cfg = EnvConfig(
        spread_scale=float(rc.spread_scale),
        ignition_rate=float(rc.ignition_rate),
        n_ignition_points=int(rc.n_ignition_points),
        randomize_ignition=True,
    )
    cfg.infra = InfraConfig(
        infra_dir=rc.infra_dir,
        catastrophe_weight=float(rc.catastrophe_weight),
        cascade_prob=float(rc.cascade_prob),
    )
    return MultiAgentWildfireEnv(state_tensor=tensor, config=cfg, num_agents=num_agents)


def _policy(family, env):
    a, g = env.action_space, env.grid_size
    if family == "nearest_fire":
        return NearestFirePolicy(a, grid_size=g, env=env)
    if family == "frontier":
        return FrontierPolicy(a, grid_size=g, env=env)
    if family == "hierarchical_greedy":
        return StrategicController(a, grid_size=g, variant="greedy_risk", env=env)
    if family == "hierarchical_risk_aware":
        return StrategicController(a, grid_size=g, variant="risk_aware", env=env)
    raise ValueError(family)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment/transfer_hybrid.yaml")
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=100000)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    regions = args.regions or list(cfg.regions.keys())
    families = args.families or list(cfg.families)
    out_dir = ensure_dir(figures_dir() / "strategic")

    for region in regions:
        rc = cfg.regions[region]
        for family in families:
            env = _env(rc, int(cfg.num_agents))
            policy = _policy(family, env)
            frames = record_rollout(env, policy, seed=args.seed, max_steps=args.steps)
            gif = out_dir / f"{region}_{family}.gif"
            render_gif(frames, gif, title=f"{region} · {family}")
            env2 = _env(rc, int(cfg.num_agents))
            film = out_dir / f"{region}_{family}_filmstrip.png"
            plot_strategic_filmstrip(
                env2, _policy(family, env2), film, seed=args.seed, max_steps=args.steps
            )
            logger.info("Wrote %s and %s (%d frames)", gif.name, film.name, len(frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
