"""Environment API + determinism tests."""

from __future__ import annotations

import numpy as np

from wildfire_rl.envs.base import WildfireEnv


def test_reset_and_step_shapes(small_tensor, env_cfg):
    env = WildfireEnv(state_tensor=small_tensor, config=env_cfg)
    obs, info = env.reset(seed=0)
    assert obs.shape == (7, 8, 8)
    assert obs.dtype == np.float32
    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (7, 8, 8)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "total_fire" in info


def test_episode_truncates_at_max_steps(small_tensor, env_cfg):
    env = WildfireEnv(state_tensor=small_tensor, config=env_cfg)
    env.reset(seed=0)
    truncated = False
    for _ in range(env_cfg.max_steps):
        _, _, terminated, truncated, _ = env.step(4)
        if terminated or truncated:
            break
    assert terminated or truncated


def test_determinism_same_seed(small_tensor, env_cfg):
    """Same seed + same actions => identical trajectories (the core seeding fix)."""
    actions = [0, 1, 2, 3, 4, 1, 1, 2, 3, 0]
    e1 = WildfireEnv(state_tensor=small_tensor, config=env_cfg)
    e2 = WildfireEnv(state_tensor=small_tensor, config=env_cfg)
    o1, _ = e1.reset(seed=123)
    o2, _ = e2.reset(seed=123)
    np.testing.assert_array_equal(o1, o2)
    for a in actions:
        s1, r1, *_ = e1.step(a)
        s2, r2, *_ = e2.step(a)
        np.testing.assert_array_equal(s1, s2)
        assert r1 == r2


def test_different_seed_differs(small_tensor):
    """Randomized ignition + different seeds => different scenarios (no leakage)."""
    from wildfire_rl.config import EnvConfig

    cfg = EnvConfig(grid_size=8, max_steps=10, randomize_ignition=True, n_ignition_points=3)
    e = WildfireEnv(state_tensor=small_tensor, config=cfg)
    o_a, _ = e.reset(seed=1)
    o_b, _ = e.reset(seed=2)
    assert not np.array_equal(o_a[0], o_b[0])


def test_gymnasium_env_checker(small_tensor, env_cfg):
    from gymnasium.utils.env_checker import check_env

    env = WildfireEnv(state_tensor=small_tensor, config=env_cfg)
    check_env(env, skip_render_check=True)
