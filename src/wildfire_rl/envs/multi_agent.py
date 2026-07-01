"""Canonical multi-agent (centralized cooperative) wildfire environment.

Replaces the two ``MultiAgentSaudiEnv`` copies. This is *centralized* cooperative
control (one policy emits a MultiDiscrete action vector for all agents) — matching the
scope described in the report. It shares the exact dynamics functions used by the
single-agent env, so there is no behavioral drift between the two.

Reward parity with single-agent env: supports ``reward_mode`` (``"raw"`` / ``"normalized"``)
and ``suppression_bonus``, matching :class:`WildfireEnv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wildfire_rl.config import EnvConfig
from wildfire_rl.envs import dynamics
from wildfire_rl.envs.base import N_ACTIONS


class MultiAgentWildfireEnv(gym.Env):
    """``num_agents`` firefighters with a shared (centralized) action vector."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        state_tensor: np.ndarray | None = None,
        tensor_path: str | Path | None = None,
        config: EnvConfig | None = None,
        num_agents: int = 3,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.num_agents = int(num_agents)

        if state_tensor is None and tensor_path is None:
            raise ValueError("Provide either `state_tensor` or `tensor_path`.")
        if state_tensor is None:
            state_tensor = np.load(tensor_path)
        self.initial_tensor = np.asarray(state_tensor, dtype=np.float32)

        c, h, w = self.initial_tensor.shape
        self.grid_size = h
        self.action_space = spaces.MultiDiscrete([N_ACTIONS] * self.num_agents)
        self._obs_channels = c + (1 if self.cfg.include_agent_channel else 0)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_channels, h, w), dtype=np.float32
        )

        self.state = self.initial_tensor.copy()
        self.agent_positions: list[list[int]] = []
        self.current_step = 0
        self._initial_fire_total = float(self.initial_tensor[0].sum()) or 1.0
        self.criticality = dynamics.load_criticality(self.cfg, self.grid_size)

    def _obs(self) -> np.ndarray:
        """State tensor + optional agent-occupancy channel (Markov observation).

        The occupancy channel marks every agent cell so the centralized policy can
        observe where its agents are — the raw ``self.state`` does not encode this.
        """
        if not self.cfg.include_agent_channel:
            return self.state.astype(np.float32)
        occ = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        for x, y in self.agent_positions:
            occ[0, x, y] += 1.0
        np.clip(occ, 0.0, 1.0, out=occ)
        return np.concatenate([self.state, occ], axis=0).astype(np.float32)

    def _init_positions(self) -> None:
        # Spread agents deterministically around the grid center.
        center = self.grid_size // 2
        offsets = [
            (0, 0),
            (-3, 0),
            (3, 0),
            (0, -3),
            (0, 3),
            (-3, -3),
            (3, 3),
            (-3, 3),
            (3, -3),
            (-6, 0),
            (6, 0),
            (0, -6),
            (0, 6),
        ]
        self.agent_positions = []
        for i in range(self.num_agents):
            dx, dy = offsets[i % len(offsets)]
            self.agent_positions.append(
                [
                    min(max(center + dx, 0), self.grid_size - 1),
                    min(max(center + dy, 0), self.grid_size - 1),
                ]
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.cfg.randomize_ignition:
            self.state = dynamics.randomize_ignition(self.initial_tensor, self.cfg, self.np_random)
        else:
            self.state = self.initial_tensor.copy()
        self._initial_fire_total = float(self.state[0].sum()) or 1.0
        self._init_positions()
        self.current_step = 0
        return self._obs(), {}

    def _move(self, idx: int, action: int) -> None:
        x, y = self.agent_positions[idx]
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.grid_size - 1, y + 1)
        self.agent_positions[idx] = [x, y]

    def step(self, actions) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        actions = np.atleast_1d(actions)
        for i in range(self.num_agents):
            self._move(i, int(actions[i]))

        dynamics.apply_suppression(self.state, [tuple(p) for p in self.agent_positions], self.cfg)
        self.state[0] = dynamics.spread_fire(self.state, self.cfg, self.np_random)
        dynamics.decay_and_deplete(self.state, self.cfg)
        dynamics.maybe_reignite(self.state, self.cfg, self.np_random)

        total_fire = float(self.state[0].sum())

        # Suppression bonus — reward agents for being near extinguished fire
        bonus = 0.0
        for pos in self.agent_positions:
            x, y = pos
            if self.state[0, x, y] < self.cfg.extinguish_threshold:
                bonus += self.cfg.suppression_bonus

        # Asset-protection penalty (petroleum criticality)
        penalty = dynamics.asset_penalty(
            self.criticality, self.state[0], self.cfg.criticality_weight
        )

        # Reward mode parity with single-agent env
        if self.cfg.reward_mode == "normalized":
            reward = float(-total_fire / self._initial_fire_total + bonus - penalty)
        else:
            reward = float(-total_fire + bonus - penalty)

        terminated = total_fire < self.cfg.termination_fire_threshold
        truncated = self.current_step >= self.cfg.max_steps
        info = {"total_fire": total_fire, "num_agents": self.num_agents}
        return self._obs(), reward, terminated, truncated, info
