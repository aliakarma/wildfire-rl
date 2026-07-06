"""Phase 15 — Saudi context enrichment: spread scaling, stochastic re-ignition,
and petroleum-asset criticality penalty."""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_rl.config import EnvConfig
from wildfire_rl.envs import dynamics
from wildfire_rl.envs.base import WildfireEnv


def _fuel_tensor(grid: int) -> np.ndarray:
    """A tensor with one central ignition and full fuel everywhere (strong spread)."""
    t = np.zeros((7, grid, grid), dtype=np.float32)
    t[0, grid // 2, grid // 2] = 1.0  # fire
    t[1, :, :] = 1.0  # fuel
    return t


def test_spread_scale_reduces_fire():
    """spread_scale < 1 must yield less fire than spread_scale = 1 (same seed/fuel)."""
    grid = 16
    t = _fuel_tensor(grid)
    common = {"grid_size": grid, "max_steps": 40, "base_spread": 0.1, "suppression_factor": 1.0}
    hi = WildfireEnv(state_tensor=t, config=EnvConfig(spread_scale=1.0, **common))
    lo = WildfireEnv(state_tensor=t, config=EnvConfig(spread_scale=0.2, **common))
    hi.reset(seed=1)
    lo.reset(seed=1)
    for _ in range(40):
        hi.step(4)  # stay
        lo.step(4)
    assert lo.state[0].sum() < hi.state[0].sum()


def test_ignition_rate_creates_new_fires():
    """ignition_rate > 0 spawns new fires even with no initial fire and no spread."""
    grid = 16
    t = np.zeros((7, grid, grid), dtype=np.float32)  # no initial fire
    cfg = EnvConfig(
        grid_size=grid,
        max_steps=50,
        ignition_rate=1.0,  # ignite every step
        ignition_intensity=1.0,
        spread_scale=0.0,  # isolate ignition from spread
        suppression_factor=1.0,  # agent cannot remove fire
    )
    env = WildfireEnv(state_tensor=t, config=cfg)
    env.reset(seed=0)
    assert env.state[0].sum() == 0.0
    for _ in range(10):
        env.step(4)
    assert env.state[0].sum() > 0.0  # spontaneous ignitions appeared


def test_criticality_penalizes_fire_on_assets():
    """Fire on a high-criticality cell must score lower than the same fire off-asset."""
    grid = 8
    t = np.zeros((7, grid, grid), dtype=np.float32)
    env = WildfireEnv(state_tensor=t, config=EnvConfig(grid_size=grid, criticality_weight=5.0))
    env.reset(seed=0)
    crit = np.zeros((grid, grid), dtype=np.float32)
    crit[2, 2] = 1.0
    env.criticality = crit

    env.state[0] = 0.0
    env.state[0, 2, 2] = 1.0  # fire on the asset
    r_asset = env._reward()

    env.state[0] = 0.0
    env.state[0, 6, 6] = 1.0  # same amount of fire, off-asset
    r_safe = env._reward()

    assert r_asset < r_safe


def test_asset_penalty_disabled_by_default():
    """No criticality map or zero weight => penalty is exactly 0."""
    fire = np.ones((4, 4), dtype=np.float32)
    crit = np.ones((4, 4), dtype=np.float32)
    assert dynamics.asset_penalty(None, fire, 5.0) == 0.0
    assert dynamics.asset_penalty(crit, fire, 0.0) == 0.0
    assert dynamics.asset_penalty(crit, fire, 2.0) == pytest.approx(2.0 * 16)


def test_load_criticality_shape_mismatch(tmp_path):
    """A raster whose shape != grid must raise."""
    p = tmp_path / "crit.npy"
    np.save(p, np.zeros((4, 4), dtype=np.float32))
    cfg = EnvConfig(criticality_path=str(p))
    with pytest.raises(ValueError):
        dynamics.load_criticality(cfg, grid_size=8)


def test_load_criticality_none_when_unset():
    assert dynamics.load_criticality(EnvConfig(), grid_size=32) is None


def test_build_criticality_raster_valid():
    """The builder produces a normalized [0,1] grid-aligned raster with a peak of 1.0."""
    import sys

    sys.path.insert(0, "scripts")
    import build_criticality as bc

    crit = bc.build_criticality(
        "saudi_eastern_province", 32, [(49.3, 25.4), (50.1, 26.4)], sigma=2.0
    )
    assert crit.shape == (32, 32)
    assert float(crit.max()) == pytest.approx(1.0)
    assert float(crit.min()) >= 0.0
