"""Ablation group + cell definitions (Phase 15B.8).

Each **cell** is a dict flipping (ideally) one factor from the full proposed hybrid system:

    {controller, low_level, variant, coordinate, catastrophe_weight, cascade_prob}

The full proposed system = hybrid risk_aware controller, coordination on, catastrophe penalty on,
cascade on. Every group holds one factor family constant and varies the ablated factor, so the
CI-separated deltas isolate that factor's causal contribution. PPO is retained only as the honest
negative baseline (referenced from the eval CSVs, not re-run here).
"""

from __future__ import annotations

import numpy as np

from wildfire_rl.config import EnvConfig, InfraConfig
from wildfire_rl.coordination.strategic_controller import StrategicController
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv
from wildfire_rl.eval.baselines import FrontierPolicy, NearestFirePolicy
from wildfire_rl.paths import region_tensor_path

GRID = 32

# Saudi petroleum region is the infrastructure test-bed for the ablations.
REGION_DIR = "saudi_eastern_province"
INFRA_DIR = "data/saudi_eastern_province/grids/32x32/infrastructure"
_DYN = {"spread_scale": 0.7, "ignition_rate": 0.05, "n_ignition_points": 6}

# Full-proposed-system defaults for a cell; each group overrides a subset.
_FULL = {
    "controller": "hybrid",
    "low_level": "frontier",
    "variant": "risk_aware",
    "coordinate": True,
    "catastrophe_weight": 5.0,
    "cascade_prob": 0.15,
}


def _cell(**overrides) -> dict:
    c = dict(_FULL)
    c.update(overrides)
    return c


GROUPS: dict[str, dict[str, dict]] = {
    # G1 — the most important ablation: hybrid vs pure heuristic (PPO = negative baseline, referenced).
    "hybrid_vs_pure": {
        "pure_heuristic_frontier": _cell(controller="heuristic", low_level="frontier"),
        "pure_heuristic_nearest": _cell(controller="heuristic", low_level="nearest_fire"),
        "hybrid_greedy": _cell(controller="hybrid", variant="greedy_risk"),
        "hybrid_risk_aware": _cell(controller="hybrid", variant="risk_aware"),  # proposed
    },
    # G2 — infrastructure-aware reward: does the objective change behavior?
    "infra_reward": {
        "no_infra_weight": _cell(catastrophe_weight=0.0, cascade_prob=0.0),
        "infra_weight": _cell(catastrophe_weight=5.0, cascade_prob=0.0),
        "cascade_risk": _cell(catastrophe_weight=5.0, cascade_prob=0.15),  # proposed
    },
    # G4 — strategic coordination component knockouts.
    "strategic_components": {
        "full": _cell(variant="risk_aware", coordinate=True),  # proposed
        "no_prioritization": _cell(variant="greedy_risk", coordinate=True),
        "no_coordination": _cell(variant="risk_aware", coordinate=False),
    },
    # G8 — catastrophic-event ablation: cascade off vs on.
    "catastrophe": {
        "no_cascade": _cell(cascade_prob=0.0),
        "cascade": _cell(cascade_prob=0.15),  # proposed
    },
}


def build_env_factory(cell: dict, num_agents: int, reward_mode: str = "normalized"):
    tensor = np.load(region_tensor_path(REGION_DIR, GRID))

    def factory():
        cfg = EnvConfig(reward_mode=reward_mode, randomize_ignition=True, **_DYN)
        cfg.infra = InfraConfig(
            infra_dir=INFRA_DIR,
            catastrophe_weight=float(cell["catastrophe_weight"]),
            cascade_prob=float(cell["cascade_prob"]),
        )
        return MultiAgentWildfireEnv(state_tensor=tensor, config=cfg, num_agents=num_agents)

    return factory


def make_cell_policy(cell: dict, sample_env):
    """Build the policy for a cell. Heuristic controller = the pure low-level router; hybrid =
    the strategic controller with the cell's variant/coordination."""
    a, g = sample_env.action_space, sample_env.grid_size
    if cell["controller"] == "heuristic":
        if cell["low_level"] == "nearest_fire":
            return NearestFirePolicy(a, grid_size=g, env=sample_env)
        return FrontierPolicy(a, grid_size=g, env=sample_env)
    # hybrid
    return StrategicController(
        a,
        grid_size=g,
        variant=cell["variant"],
        low_level=cell["low_level"],
        coordinate=bool(cell["coordinate"]),
        env=sample_env,
    )
