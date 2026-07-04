"""Phase-1 integration smoke tests for the Cell2Fire Gymnasium env.

These need the interactive Cell2Fire binary built on Linux
(``third_party/firehose/cell2fire/Cell2FireC/Cell2Fire``); they skip cleanly on a bare
checkout (e.g. CI without the C++ build) so the ported-core test suite stays green.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from wildfire_marl.env.cell2fire_binding import DEFAULT_BINARY

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="Cell2Fire runs on Linux only"),
    pytest.mark.skipif(
        not DEFAULT_BINARY.exists(), reason="interactive Cell2Fire binary not built"
    ),
]


def _make_env(**kwargs):
    from wildfire_marl.env.single_agent_env import FireSuppressionEnv

    kwargs.setdefault("fire_map", "Sub20x20")  # smallest stock map -> fastest tests
    kwargs.setdefault("steps_before_sim", 10)
    return FireSuppressionEnv(**kwargs)


def test_env_steps_end_to_end():
    env = _make_env(render_mode="rgb_array")
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert info["cells_on_fire"] >= 0
        obs, reward, terminated, truncated, info = env.step(int(env.action_space.sample()))
        assert env.observation_space.contains(obs)
        assert reward <= 0.0  # FireSizeReward is non-positive
        frame = env.render()
        assert frame.dtype == np.uint8 and frame.shape == (env.height, env.width, 3)
    finally:
        env.close()


def test_harvest_registers_in_observation():
    env = _make_env()
    try:
        env.reset(seed=1)
        # Treat a fuel cell far from everything; it must show up in the harvested channel.
        cell = int(np.flatnonzero(env.fuel_mask.ravel() > 0)[0])
        obs, *_ = env.step(cell)
        y, x = divmod(cell, env.width)
        assert obs[1, y, x] == 1.0, "harvested cell not reflected in observation"
    finally:
        env.close()


def test_deterministic_given_seed():
    def rollout(seed: int):
        env = _make_env()
        try:
            obs, info = env.reset(seed=seed)
            traj, rews = [obs.copy()], []
            for a in (5, 105, 205, 305):
                obs, r, term, trunc, _ = env.step(a)
                traj.append(obs.copy())
                rews.append(r)
                if term or trunc:
                    break
            return info["ignition_cell"], traj, rews
        finally:
            env.close()

    ig1, t1, r1 = rollout(7)
    ig2, t2, r2 = rollout(7)
    assert ig1 == ig2
    assert r1 == r2
    assert all(np.array_equal(a, b) for a, b in zip(t1, t2, strict=True))
