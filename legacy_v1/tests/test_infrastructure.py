"""Critical / petroleum-infrastructure modeling (Phase 15B.1).

Covers the pure dynamics (cascade detonation, catastrophe penalty, raster loading) and the env
integration (infrastructure observation channel, catastrophe reward). All infra features default to
no-op, so this file also asserts backward compatibility.
"""

from __future__ import annotations

import numpy as np

from wildfire_rl.config import EnvConfig, InfraConfig
from wildfire_rl.envs import dynamics
from wildfire_rl.envs.base import WildfireEnv
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv


def _write_infra(tmp_path, grid=8):
    """Create a tiny infra raster set: a refinery (1) and a pipeline (2)."""
    at = np.zeros((grid, grid), dtype=np.int32)
    at[2, 2] = 1  # refinery
    at[5, 5] = 2  # pipeline
    crit = np.zeros((grid, grid), dtype=np.float32)
    crit[2, 2] = 1.0
    crit[5, 5] = 0.4
    np.save(tmp_path / "asset_type.npy", at)
    np.save(tmp_path / "criticality.npy", crit)
    return at, crit


# --------------------------------------------------------------------- config / backward-compat
def test_infra_config_defaults_are_noop():
    cfg = EnvConfig()
    assert cfg.infra.infra_dir is None
    assert cfg.infra.observe_infra is False
    assert cfg.infra.cascade_prob == 0.0
    assert cfg.infra.catastrophe_weight == 0.0


def test_obs_unchanged_without_infra(small_tensor, env_cfg):
    env = WildfireEnv(state_tensor=small_tensor, config=env_cfg)
    obs, _ = env.reset(seed=0)
    # 7 state channels + 1 agent channel, no infra channel
    assert obs.shape[0] == small_tensor.shape[0] + 1


# ------------------------------------------------------------------------------ pure dynamics
def test_cascade_explosion_ignites_within_radius():
    grid = 8
    at = np.zeros((grid, grid), dtype=np.int32)
    at[4, 4] = 1  # a refinery
    state = np.zeros((7, grid, grid), dtype=np.float32)
    state[0, 4, 4] = 1.0  # refinery is burning
    infra = InfraConfig(cascade_prob=1.0, blast_radius=2)  # deterministic full cascade
    rng = np.random.default_rng(0)
    n = dynamics.cascade_explosion(state, at, infra, rng)
    assert n > 0
    assert state[0, 4, 5] == 1.0  # a neighbor inside the blast radius ignited
    assert state[0, 4, 7] == 0.0  # outside the radius stays clean


def test_cascade_noop_when_disabled():
    grid = 8
    at = np.zeros((grid, grid), dtype=np.int32)
    at[4, 4] = 1
    state = np.zeros((7, grid, grid), dtype=np.float32)
    state[0, 4, 4] = 1.0
    infra = InfraConfig(cascade_prob=0.0)  # disabled
    assert dynamics.cascade_explosion(state, at, infra, np.random.default_rng(0)) == 0
    assert (
        dynamics.cascade_explosion(
            state, None, InfraConfig(cascade_prob=1.0), np.random.default_rng(0)
        )
        == 0
    )


def test_catastrophe_penalty_orders_by_asset_value():
    grid = 8
    at = np.zeros((grid, grid), dtype=np.int32)
    at[2, 2] = 1  # refinery (value 10)
    at[5, 5] = 2  # pipeline (value 4)
    values = {1: 10.0, 2: 4.0}

    fire_ref = np.zeros((grid, grid), dtype=np.float32)
    fire_ref[2, 2] = 1.0
    fire_pipe = np.zeros((grid, grid), dtype=np.float32)
    fire_pipe[5, 5] = 1.0

    p_ref = dynamics.catastrophe_penalty(at, fire_ref, values, weight=1.0)
    p_pipe = dynamics.catastrophe_penalty(at, fire_pipe, values, weight=1.0)
    assert p_ref > p_pipe > 0.0
    # disabled -> 0
    assert dynamics.catastrophe_penalty(at, fire_ref, values, weight=0.0) == 0.0
    assert dynamics.catastrophe_penalty(None, fire_ref, values, weight=1.0) == 0.0


def test_load_infrastructure_shape_validation(tmp_path):
    _write_infra(tmp_path, grid=8)
    cfg = EnvConfig()
    cfg.infra = InfraConfig(infra_dir=str(tmp_path))
    at, crit = dynamics.load_infrastructure(cfg, grid_size=8)
    assert at is not None and at.shape == (8, 8)
    assert crit is not None and crit.shape == (8, 8)
    # wrong grid size raises
    import pytest

    with pytest.raises(ValueError):
        dynamics.load_infrastructure(cfg, grid_size=16)


# --------------------------------------------------------------------------- env integration
def test_infra_observation_channel(tmp_path, small_tensor):
    _write_infra(tmp_path, grid=small_tensor.shape[1])
    cfg = EnvConfig()
    cfg.infra = InfraConfig(infra_dir=str(tmp_path), observe_infra=True)
    env = WildfireEnv(state_tensor=small_tensor, config=cfg)
    obs, _ = env.reset(seed=0)
    # 7 state + 1 agent + 1 infra-criticality
    assert obs.shape[0] == small_tensor.shape[0] + 2
    assert env.observation_space.shape[0] == obs.shape[0]


def test_catastrophe_reward_penalizes_fire_on_assets(tmp_path, small_tensor):
    at, _ = _write_infra(tmp_path, grid=small_tensor.shape[1])
    cfg = EnvConfig()
    cfg.infra = InfraConfig(infra_dir=str(tmp_path), catastrophe_weight=5.0)
    clean = np.zeros_like(small_tensor)

    env_ref = WildfireEnv(state_tensor=clean, config=cfg)
    env_ref.reset(seed=1)
    env_ref.state[0, 2, 2] = 1.0  # fire on the refinery
    r_ref = env_ref._reward()

    env_empty = WildfireEnv(state_tensor=clean, config=cfg)
    env_empty.reset(seed=1)
    env_empty.state[0, 0, 0] = 1.0  # fire on an empty cell
    r_empty = env_empty._reward()

    assert r_ref < r_empty


def test_marl_infra_channel_and_cascade(tmp_path, small_tensor):
    _write_infra(tmp_path, grid=small_tensor.shape[1])
    cfg = EnvConfig()
    cfg.infra = InfraConfig(
        infra_dir=str(tmp_path), observe_infra=True, cascade_prob=1.0, blast_radius=2
    )
    env = MultiAgentWildfireEnv(state_tensor=small_tensor, config=cfg, num_agents=3)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == small_tensor.shape[0] + 2  # state + occupancy + infra
    env.state[0, 2, 2] = 1.0  # ignite the refinery
    _, _, _, _, info = env.step([4, 4, 4])
    assert info["cascade_ignited"] >= 0  # cascade key present and runs
