"""Environment API + determinism tests."""

from __future__ import annotations

import numpy as np

from wildfire_rl.envs.base import WildfireEnv


def test_reset_and_step_shapes(small_tensor, env_cfg):
    env = WildfireEnv(state_tensor=small_tensor, config=env_cfg)
    obs, info = env.reset(seed=0)
    assert obs.shape == (8, 8, 8)  # 7 state channels + 1 agent-position channel
    assert obs.dtype == np.float32
    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (8, 8, 8)  # 7 state channels + 1 agent-position channel
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


def test_observation_encodes_agent_position(small_tensor, env_cfg):
    """The Markov fix: obs carries an agent-position channel that changes when the agent moves."""
    env = WildfireEnv(state_tensor=small_tensor, config=env_cfg)
    o0, _ = env.reset(seed=0)
    # One extra channel appended to the 7 state channels.
    assert o0.shape[0] == small_tensor.shape[0] + 1
    o1, *_ = env.step(1)  # action 1 = move down -> position channel must change
    assert not np.array_equal(o0[-1], o1[-1]), "agent-position channel did not change after a move"
    # The state channels themselves are still exposed at the front.
    np.testing.assert_array_equal(o0[: small_tensor.shape[0]].shape, small_tensor.shape)


def test_include_agent_channel_toggle(small_tensor):
    """With include_agent_channel=False the observation reverts to the raw state shape."""
    from wildfire_rl.config import EnvConfig

    cfg = EnvConfig(grid_size=8, max_steps=10, include_agent_channel=False)
    env = WildfireEnv(state_tensor=small_tensor, config=cfg)
    o0, _ = env.reset(seed=0)
    assert o0.shape == (7, 8, 8)


def test_agent_suppression_reward_credits_removal():
    """Phase 16: a step that removes fire scores higher when the suppression weight is on."""
    from wildfire_rl.config import EnvConfig

    t = np.zeros((7, 8, 8), dtype=np.float32)
    t[0, 4, 4] = 1.0  # fire directly under the agent's start position [4, 4]

    def reward_after_suppress(weight: float) -> float:
        cfg = EnvConfig(grid_size=8, max_steps=10, reward_agent_suppression_weight=weight)
        env = WildfireEnv(state_tensor=t, config=cfg)
        env.reset(seed=0)
        _, r, *_ = env.step(4)  # stay -> suppress the fire at [4, 4]
        return r

    assert reward_after_suppress(10.0) > reward_after_suppress(0.0)
