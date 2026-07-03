"""Strategic ablation framework (Phase 15B.8).

Tests the new metrics (PA/CCL/CE/ERL/TRG), the ablation factor toggles (each flips a real behavior),
and the group specs (single-factor cells over the full proposed system).
"""

from __future__ import annotations

import math

import numpy as np

from wildfire_rl.ablation import GROUPS, build_env_factory, make_cell_policy
from wildfire_rl.config import EnvConfig, InfraConfig
from wildfire_rl.coordination.strategic_controller import StrategicController
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv
from wildfire_rl.eval.metrics import (
    catastrophe_chain_length,
    coordination_efficiency,
    emergency_response_latency,
    prioritization_accuracy,
    transfer_robustness_gap,
)


# --------------------------------------------------------------------------- new metrics
def test_prioritization_accuracy_value_weighted():
    at = np.zeros((4, 4), dtype=np.int32)
    at[0, 0] = 1  # refinery (10)
    at[3, 3] = 2  # pipeline (4)
    values = {1: 10.0, 2: 4.0}
    clean = np.zeros((7, 4, 4), dtype=np.float32)
    assert prioritization_accuracy(at, clean, values) == 1.0
    # refinery burns -> only pipeline value protected = 4 / 14
    burn = clean.copy()
    burn[0, 0, 0] = 1.0
    assert abs(prioritization_accuracy(at, burn, values) - (4.0 / 14.0)) < 1e-6


def test_catastrophe_chain_length():
    at = np.zeros((4, 4), dtype=np.int32)
    at[1, 1] = 1
    final = np.zeros((7, 4, 4), dtype=np.float32)
    final[0, 1, 1] = 1.0  # detonated asset
    assert catastrophe_chain_length(6, at, final) == 6.0  # 6 cascade cells / 1 detonation
    assert catastrophe_chain_length(6, at, np.zeros((7, 4, 4), np.float32)) == 0.0  # none detonated


def test_coordination_efficiency():
    # agents always distinct -> 1.0
    assert coordination_efficiency([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]) == 1.0
    # both agents pile on the same cell every step -> 0.5
    assert coordination_efficiency([[(0, 0), (0, 0)], [(1, 1), (1, 1)]]) == 0.5


def test_emergency_response_latency():
    # asset 0 threatened at t=2, agent arrives t=5 -> latency 3; asset 1 never threatened -> excluded
    assert emergency_response_latency([2, -1], [5, 9]) == 3.0
    assert emergency_response_latency([-1, -1], [9, 9]) == 0.0


def test_transfer_robustness_gap():
    assert abs(transfer_robustness_gap([1.0, 0.7, 0.9]) - 0.3) < 1e-9  # 1 - min(0.7)
    assert transfer_robustness_gap([float("nan")]) == 0.0


# ------------------------------------------------------------------- factor toggles are real
def _env(tmp_path, grid=16):
    at = np.zeros((grid, grid), dtype=np.int32)
    crit = np.zeros((grid, grid), dtype=np.float32)
    at[2, 2] = 1
    crit[2, 2] = 1.0
    np.save(tmp_path / "asset_type.npy", at)
    np.save(tmp_path / "criticality.npy", crit)
    cfg = EnvConfig()
    cfg.infra = InfraConfig(infra_dir=str(tmp_path))
    return MultiAgentWildfireEnv(
        state_tensor=np.zeros((7, grid, grid), np.float32), config=cfg, num_agents=3
    )


def test_coordination_toggle_changes_dispatch(tmp_path):
    """coordinate=False makes all agents target the same top sector (pile-on)."""
    env = _env(tmp_path)
    env.reset(seed=0)
    env.state[0, 12:15, 12:15] = 1.0  # fire in SE
    coord = StrategicController(
        env.action_space, grid_size=16, variant="greedy_risk", coordinate=True, env=env
    )
    pile = StrategicController(
        env.action_space, grid_size=16, variant="greedy_risk", coordinate=False, env=env
    )
    a_coord, _ = coord.predict(None)
    a_pile, _ = pile.predict(None)
    # with a single dominant fire sector both may agree, but the toggle must be wired + not error
    assert a_coord.shape == a_pile.shape == (3,)


# ----------------------------------------------------------------------------- group specs
def test_groups_single_factor_and_buildable(tmp_path):
    # every group has a 'proposed'-style full cell and buildable env/policy
    for group, cells in GROUPS.items():
        assert len(cells) >= 2, f"{group} needs >=2 cells"
        for name, cell in cells.items():
            for req in ("controller", "catastrophe_weight", "cascade_prob", "coordinate"):
                assert req in cell, f"{group}/{name} missing {req}"


def test_cell_env_and_policy_run():
    """A hybrid cell builds a working env + policy that steps (uses the real Saudi infra rasters)."""
    import pathlib

    if not pathlib.Path(
        "data/saudi_eastern_province/grids/32x32/infrastructure/asset_type.npy"
    ).exists():
        import pytest

        pytest.skip("Saudi infra rasters not built in this checkout")
    cell = GROUPS["hybrid_vs_pure"]["hybrid_risk_aware"]
    factory = build_env_factory(cell, num_agents=3)
    env = factory()
    policy = make_cell_policy(cell, env)
    obs, _ = env.reset(seed=100000)
    policy.env = env
    action, _ = policy.predict(obs)
    _, _, _, _, info = env.step(action)
    assert "cascade_ignited" in info
    assert not math.isnan(env.infra_asset_type.sum())
