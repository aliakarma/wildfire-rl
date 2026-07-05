"""Unit tests for Phase 4 baseline agents and evaluation loop."""

from __future__ import annotations

import numpy as np

from wildfire_marl.agents.heuristics import (
    GreatestRiskFirstPolicy,
    NearestFrontierPolicy,
    NoOpPolicy,
    RandomPolicy,
    ValueWeightedFirstPolicy,
)
from wildfire_marl.eval.evaluate import evaluate_policy


def test_heuristics_predictions():
    """Verify that all heuristics predict valid actions on mock observations."""
    h, w = 8, 8
    # 4-channel obs (with criticality channel)
    obs = np.zeros((4, h, w), dtype=np.float32)

    # Setup fuel mask (channel 2)
    obs[2] = 1.0  # all cells are fuel

    # Place fire (channel 0) on cell (3, 3)
    obs[0, 3, 3] = 1.0

    # Set criticality (channel 3) with a peak at (3, 5)
    obs[3, 3, 5] = 1.0

    policies = [
        NoOpPolicy(),
        RandomPolicy(),
        NearestFrontierPolicy(),
        GreatestRiskFirstPolicy(),
        ValueWeightedFirstPolicy(),
    ]

    for policy in policies:
        action, state = policy.predict(obs)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < h * w
        assert state is None


class DummyPolicy:
    """Mock policy that returns 0 action."""

    def predict(
        self, obs: np.ndarray, action_masks: np.ndarray | None = None, deterministic: bool = True
    ) -> tuple[int, None]:
        return 0, None


class DummyEnv:
    """Mock environment that mimics FireSuppressionEnv without spawning Cell2Fire."""

    def __init__(self):
        self.height = 8
        self.width = 8
        self.num_cells = 64
        self.fuel_mask = np.ones((8, 8), dtype=np.float32)
        self.fire_state = np.zeros((8, 8), dtype=np.int8)
        self.fire_state[3, 3] = 1  # active fire at start
        self.ignition_cell = 27
        self.binding = type("DummyBinding", (object,), {"finished": False})()
        self.iter = 0
        self.max_steps = 5
        self.asset_type = np.zeros((8, 8), dtype=np.int32)
        self.asset_type[1, 1] = 1  # Refinery
        self.asset_values = {1: 10.0}
        self.cascade_ignited_count = 0
        self.criticality = np.zeros((8, 8), dtype=np.float32)
        self.previously_detonated = set()

    def reset(self, seed=None):
        self.iter = 0
        self.fire_state = np.zeros((8, 8), dtype=np.int8)
        self.fire_state[3, 3] = 1
        return self._obs(), {}

    def step(self, action):
        self.iter += 1
        self.fire_state[0, 0] = -1  # harvested
        terminated = False
        truncated = self.iter >= self.max_steps
        return self._obs(), -1.0, terminated, truncated, {}

    def _obs(self):
        obs = np.zeros((4, 8, 8), dtype=np.float32)
        obs[0] = self.fire_state > 0
        obs[1] = self.fire_state < 0
        obs[2] = self.fuel_mask
        obs[3] = self.criticality
        return obs

    def close(self):
        pass


def test_evaluate_policy_loop():
    """Verify that evaluate_policy loop works end-to-end and returns correct metrics."""

    def env_factory():
        return DummyEnv()

    policy = DummyPolicy()

    res = evaluate_policy(
        policy=policy,
        env_factory=env_factory,
        n_episodes=2,
        scenario_seed_offset=0,
    )

    assert "summary" in res
    assert "episodes" in res
    assert len(res["episodes"]) == 2

    # Check metric keys
    ep_keys = res["episodes"][0].keys()
    assert "episode_reward" in ep_keys
    assert "burned_cells" in ep_keys
    assert "fire_intensity" in ep_keys
    assert "containment_rate" in ep_keys
    assert "isr" in ep_keys
    assert "pca" in ep_keys
    assert "wel" in ep_keys
    assert "cps" in ep_keys

    summary_keys = res["summary"].keys()
    assert "episode_reward_mean" in summary_keys
    assert "episode_reward_std" in summary_keys
    assert "burned_cells_mean" in summary_keys
    assert res["summary"]["n_episodes"] == 2.0
