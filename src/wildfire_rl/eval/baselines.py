"""Baseline control policies.

The original "baseline_evaluation" notebook contained NO baseline — only the PPO model
evaluated on its own region. Without a control there is no evidence the learned policy
beats doing nothing. These policies expose the same ``predict(obs, deterministic)``
interface as Stable-Baselines3 models, so they drop straight into
:func:`wildfire_rl.eval.evaluate.evaluate_policy`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Action 4 == "stay" (no movement); see wildfire_rl.envs.base.
STAY_ACTION = 4


class RandomPolicy:
    """Uniformly random actions. The lower-bound control for 'did the agent learn?'."""

    uses_privileged_state = False

    def __init__(self, action_space, seed: int | None = None) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = True):  # noqa: ARG002
        if hasattr(self.action_space, "nvec"):  # MultiDiscrete (MARL)
            action = np.array([self.rng.integers(0, n) for n in self.action_space.nvec])
        else:
            action = int(self.rng.integers(0, self.action_space.n))
        return action, None


class NoOpPolicy:
    """Agent never moves (constant 'stay'). Note: local suppression still applies."""

    uses_privileged_state = False

    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def predict(self, obs, deterministic: bool = True):  # noqa: ARG002
        if hasattr(self.action_space, "nvec"):
            action = np.full(len(self.action_space.nvec), STAY_ACTION, dtype=int)
        else:
            action = STAY_ACTION
        return action, None


class NearestFirePolicy:
    """Agent moves towards the closest burning cell.

    If no fire exists, it stays. Supports both single-agent and MultiDiscrete MARL.

    Privileged baseline: reads ``env.agent_pos`` / ``env.agent_positions`` directly for oracle
    self-localization (see ``uses_privileged_state``). PPO does not receive the fire-argmin and
    localizes only via its observation channel (Phase 3), so any heuristic advantage must be
    reported with this asymmetry stated explicitly.
    """

    #: reads the agent's true grid position from the env (oracle localization)
    uses_privileged_state = True

    def __init__(self, action_space, grid_size: int = 32, env: Any = None) -> None:
        self.action_space = action_space
        self.grid_size = grid_size
        self.env = env
        self.agent_pos = [grid_size // 2, grid_size // 2]

    def predict(self, obs, deterministic: bool = True):  # noqa: ARG002
        fire_channel = obs[0]
        burning_indices = np.argwhere(fire_channel > 0.1)

        if hasattr(self.action_space, "nvec"):
            num_agents = len(self.action_space.nvec)
            if self.env is not None and hasattr(self.env, "agent_positions"):
                agent_positions = [list(pos) for pos in self.env.agent_positions]
            else:
                center = self.grid_size // 2
                offsets = [(0, 0), (-3, 0), (3, 0), (0, -3), (0, 3), (-3, -3), (3, 3)]
                agent_positions = []
                for i in range(num_agents):
                    dx, dy = offsets[i % len(offsets)]
                    agent_positions.append([
                        min(max(center + dx, 0), self.grid_size - 1),
                        min(max(center + dy, 0), self.grid_size - 1)
                    ])

            if len(burning_indices) == 0:
                return np.full(num_agents, STAY_ACTION, dtype=int), None

            actions = []
            for ax, ay in agent_positions:
                dists = np.abs(burning_indices[:, 0] - ax) + np.abs(burning_indices[:, 1] - ay)
                closest_idx = np.argmin(dists)
                target_x, target_y = burning_indices[closest_idx]

                dx = target_x - ax
                dy = target_y - ay

                if abs(dx) >= abs(dy) and dx != 0:
                    action = 0 if dx < 0 else 1
                elif dy != 0:
                    action = 2 if dy < 0 else 3
                else:
                    action = STAY_ACTION
                actions.append(action)
            return np.array(actions, dtype=int), None
        else:
            if self.env is not None and hasattr(self.env, "agent_pos"):
                self.agent_pos = list(self.env.agent_pos)

            if len(burning_indices) == 0:
                return STAY_ACTION, None

            ax, ay = self.agent_pos
            dists = np.abs(burning_indices[:, 0] - ax) + np.abs(burning_indices[:, 1] - ay)
            closest_idx = np.argmin(dists)
            target_x, target_y = burning_indices[closest_idx]

            dx = target_x - ax
            dy = target_y - ay

            if abs(dx) >= abs(dy) and dx != 0:
                action = 0 if dx < 0 else 1
            elif dy != 0:
                action = 2 if dy < 0 else 3
            else:
                action = STAY_ACTION

            if action == 0:
                self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
            elif action == 1:
                self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
            elif action == 2:
                self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
            elif action == 3:
                self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)

            return action, None


class FrontierPolicy:
    """Agent moves towards the closest fire frontier cell (burning but has unburnt neighbors).

    Privileged baseline: reads ``env.agent_pos`` / ``env.agent_positions`` directly for oracle
    self-localization (see ``uses_privileged_state``). Reported with the same asymmetry caveat as
    :class:`NearestFirePolicy`.
    """

    #: reads the agent's true grid position from the env (oracle localization)
    uses_privileged_state = True

    def __init__(self, action_space, grid_size: int = 32, env: Any = None) -> None:
        self.action_space = action_space
        self.grid_size = grid_size
        self.env = env
        self.agent_pos = [grid_size // 2, grid_size // 2]

    def predict(self, obs, deterministic: bool = True):  # noqa: ARG002
        fire_channel = obs[0]
        active = fire_channel > 0.1
        h, w = fire_channel.shape

        # Count active neighbors
        from scipy.ndimage import convolve
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        active_neighbors = convolve(active.astype(int), kernel, mode='constant', cval=0)

        # Frontier cells are active but have < 4 active neighbors
        frontier = active & (active_neighbors < 4)
        burning_indices = np.argwhere(frontier)

        if len(burning_indices) == 0:
            # Fall back to nearest fire cell
            burning_indices = np.argwhere(active)

        if hasattr(self.action_space, "nvec"):
            num_agents = len(self.action_space.nvec)
            if self.env is not None and hasattr(self.env, "agent_positions"):
                agent_positions = [list(pos) for pos in self.env.agent_positions]
            else:
                center = self.grid_size // 2
                offsets = [(0, 0), (-3, 0), (3, 0), (0, -3), (0, 3), (-3, -3), (3, 3)]
                agent_positions = []
                for i in range(num_agents):
                    dx, dy = offsets[i % len(offsets)]
                    agent_positions.append([
                        min(max(center + dx, 0), self.grid_size - 1),
                        min(max(center + dy, 0), self.grid_size - 1)
                    ])

            if len(burning_indices) == 0:
                return np.full(num_agents, STAY_ACTION, dtype=int), None

            actions = []
            for ax, ay in agent_positions:
                dists = np.abs(burning_indices[:, 0] - ax) + np.abs(burning_indices[:, 1] - ay)
                closest_idx = np.argmin(dists)
                target_x, target_y = burning_indices[closest_idx]

                dx = target_x - ax
                dy = target_y - ay

                if abs(dx) >= abs(dy) and dx != 0:
                    action = 0 if dx < 0 else 1
                elif dy != 0:
                    action = 2 if dy < 0 else 3
                else:
                    action = STAY_ACTION
                actions.append(action)
            return np.array(actions, dtype=int), None
        else:
            if self.env is not None and hasattr(self.env, "agent_pos"):
                self.agent_pos = list(self.env.agent_pos)

            if len(burning_indices) == 0:
                return STAY_ACTION, None

            ax, ay = self.agent_pos
            dists = np.abs(burning_indices[:, 0] - ax) + np.abs(burning_indices[:, 1] - ay)
            closest_idx = np.argmin(dists)
            target_x, target_y = burning_indices[closest_idx]

            dx = target_x - ax
            dy = target_y - ay

            if abs(dx) >= abs(dy) and dx != 0:
                action = 0 if dx < 0 else 1
            elif dy != 0:
                action = 2 if dy < 0 else 3
            else:
                action = STAY_ACTION

            if action == 0:
                self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
            elif action == 1:
                self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
            elif action == 2:
                self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
            elif action == 3:
                self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)

            return action, None
