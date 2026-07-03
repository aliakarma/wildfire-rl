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

        # Critical-infrastructure layers (Phase 15B.1): asset-type grid + criticality raster.
        self.infra_asset_type, self.infra_criticality = dynamics.load_infrastructure(
            self.cfg, self.grid_size
        )
        self._observe_infra = bool(
            self.cfg.infra.observe_infra and self.infra_criticality is not None
        )

        self._obs_channels = (
            c + (1 if self.cfg.include_agent_channel else 0) + (1 if self._observe_infra else 0)
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_channels, h, w), dtype=np.float32
        )

        self.state = self.initial_tensor.copy()
        self.agent_pos = [h // 2, w // 2]
        self.current_step = 0
        self._initial_fire_total = float(self.initial_tensor[0].sum()) or 1.0
        self.criticality = dynamics.load_criticality(self.cfg, self.grid_size)
        self._cascade_ignited_total = 0

    # -------------------------------------------------------------- observation
    def _obs(self) -> np.ndarray:
        """Policy observation: state tensor + optional agent-position channel.

        The agent-position channel is what makes the environment Markov for a
        movement policy — the raw ``self.state`` alone does not encode where the
        agent is, so without this the policy cannot learn to navigate to the fire.
        """
        layers = [self.state]
        if self.cfg.include_agent_channel:
            pos = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
            x, y = self.agent_pos
            pos[0, x, y] = 1.0
            layers.append(pos)
        # Infrastructure-criticality channel (Phase 15B.1): strategic situational awareness.
        if self._observe_infra:
            layers.append(self.infra_criticality[None, :, :])
        if len(layers) == 1:
            return self.state.astype(np.float32)
        return np.concatenate(layers, axis=0).astype(np.float32)

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
        self._cascade_ignited_total = 0
        return self._obs(), {}

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

        fire_before = self.state[0].copy()
        dynamics.apply_suppression(self.state, [tuple(self.agent_pos)], self.cfg)
        agent_removed = float((fire_before - self.state[0]).sum())
        self.state[0] = dynamics.spread_fire(self.state, self.cfg, self.np_random)
        dynamics.decay_and_deplete(self.state, self.cfg)
        dynamics.maybe_reignite(self.state, self.cfg, self.np_random)
        # Cascading petroleum detonation (Phase 15B.1): burning assets ignite their blast radius.
        self._cascade_ignited_total += dynamics.cascade_explosion(
            self.state, self.infra_asset_type, self.cfg.infra, self.np_random
        )

        reward = self._reward(agent_removed)
        total_fire = float(self.state[0].sum())
        terminated = total_fire < self.cfg.termination_fire_threshold
        truncated = self.current_step >= self.cfg.max_steps
        info = {
            "total_fire": total_fire,
            "step": self.current_step,
            "cascade_ignited": self._cascade_ignited_total,
        }
        return self._obs(), reward, terminated, truncated, info

    def _reward(self, agent_removed: float = 0.0) -> float:
        total_fire = float(self.state[0].sum())
        x, y = self.agent_pos
        bonus = (
            self.cfg.suppression_bonus
            if self.state[0, x, y] < self.cfg.suppression_bonus_threshold
            else 0.0
        )
        penalty = dynamics.asset_penalty(
            self.criticality, self.state[0], self.cfg.criticality_weight
        )
        # Catastrophe penalty (Phase 15B.1): value-weighted cost of fire reaching infrastructure,
        # so the agent defends high-value assets (refineries) rather than only minimizing burned area.
        penalty += dynamics.catastrophe_penalty(
            self.infra_asset_type,
            self.state[0],
            self.cfg.infra.asset_values,
            self.cfg.infra.catastrophe_weight,
        )
        # Agent-attributable suppression credit: fire the agent actually removed this step,
        # normalized by initial fire — a learnable signal tied to the agent's own actions.
        supp = self.cfg.reward_agent_suppression_weight * agent_removed / self._initial_fire_total
        prox = self._proximity_reward()
        fw = self.cfg.reward_fire_weight
        if self.cfg.reward_mode == "normalized":
            return float(
                -fw * total_fire / self._initial_fire_total + bonus - penalty + supp + prox
            )
        return float(-fw * total_fire + bonus - penalty + supp + prox)

    def _proximity_reward(self) -> float:
        """Dense guidance: reward for being near the nearest burning cell (0 when disabled)."""
        w = self.cfg.reward_proximity_weight
        if w <= 0.0:
            return 0.0
        fire = np.argwhere(self.state[0] > self.cfg.spread_threshold)
        if len(fire) == 0:
            return 0.0
        ax, ay = self.agent_pos
        dmin = int((np.abs(fire[:, 0] - ax) + np.abs(fire[:, 1] - ay)).min())
        return float(w * max(0.0, 1.0 - dmin / self.grid_size))


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
