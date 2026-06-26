"""Canonical single-agent wildfire environment.

Replaces ``SaudiWildfireEnv``, ``CaliforniaWildfireEnv``, ``SaudiEvaluationEnv`` and
``CaliforniaEvaluationEnv`` (four byte-identical copies). The region is now just a
state tensor passed in; dynamics constants come from :class:`EnvConfig`; stochasticity
uses ``self.np_random`` so episodes are reproducible.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wildfire_rl.config import EnvConfig
from wildfire_rl.envs import dynamics

# Discrete actions: 0 up, 1 down, 2 left, 3 right, 4 stay (suppress in place).
N_ACTIONS = 5


class WildfireEnv(gym.Env):
    """A single firefighting agent suppressing fire on a geospatial state tensor."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        state_tensor: np.ndarray | None = None,
        tensor_path: str | Path | None = None,
        config: EnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()

        if state_tensor is None and tensor_path is None:
            raise ValueError("Provide either `state_tensor` or `tensor_path`.")
        if state_tensor is None:
            state_tensor = np.load(tensor_path)
        self.initial_tensor = np.asarray(state_tensor, dtype=np.float32)

        c, h, w = self.initial_tensor.shape
        if c != self.cfg.n_channels:
            raise ValueError(f"Expected {self.cfg.n_channels} channels, got {c}")
        self.grid_size = h

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )

        self.state = self.initial_tensor.copy()
        self.agent_pos = [h // 2, w // 2]
        self.current_step = 0
        self._initial_fire_total = float(self.initial_tensor[0].sum()) or 1.0

    # ------------------------------------------------------------------ reset
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)  # seeds self.np_random reproducibly
        if self.cfg.randomize_ignition:
            self.state = dynamics.randomize_ignition(self.initial_tensor, self.cfg, self.np_random)
        else:
            self.state = self.initial_tensor.copy()
        self._initial_fire_total = float(self.state[0].sum()) or 1.0
        self.agent_pos = [self.grid_size // 2, self.grid_size // 2]
        self.current_step = 0
        return self.state.astype(np.float32), {}

    # ------------------------------------------------------------------- step
    def _move(self, action: int) -> None:
        x, y = self.agent_pos
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.grid_size - 1, y + 1)
        # action == 4: stay
        self.agent_pos = [x, y]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        self._move(int(action))

        dynamics.apply_suppression(self.state, [tuple(self.agent_pos)], self.cfg)
        self.state[0] = dynamics.spread_fire(self.state, self.cfg, self.np_random)
        dynamics.decay_and_deplete(self.state, self.cfg)

        reward = self._reward()
        total_fire = float(self.state[0].sum())
        terminated = total_fire < self.cfg.termination_fire_threshold
        truncated = self.current_step >= self.cfg.max_steps
        info = {"total_fire": total_fire, "step": self.current_step}
        return self.state.astype(np.float32), reward, terminated, truncated, info

    def _reward(self) -> float:
        total_fire = float(self.state[0].sum())
        x, y = self.agent_pos
        bonus = (
            self.cfg.suppression_bonus
            if self.state[0, x, y] < self.cfg.suppression_bonus_threshold
            else 0.0
        )
        if self.cfg.reward_mode == "normalized":
            return float(-total_fire / self._initial_fire_total + bonus)
        return float(-total_fire + bonus)


def make_env_factory(
    state_tensor: np.ndarray | None = None,
    tensor_path: str | Path | None = None,
    config: EnvConfig | None = None,
) -> Callable[[], WildfireEnv]:
    """Return a zero-arg factory building a fresh env (for SB3 vec-envs / evaluation)."""
    # Load once; each env gets its own copy of the array.
    if state_tensor is None and tensor_path is not None:
        state_tensor = np.load(tensor_path)

    def _factory() -> WildfireEnv:
        return WildfireEnv(state_tensor=state_tensor, config=config)

    return _factory
