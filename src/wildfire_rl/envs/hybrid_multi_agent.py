"""Hybrid Path-Planning + RL Coordination multi-agent environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from wildfire_rl.config import EnvConfig
from wildfire_rl.envs.multi_agent_v3 import MultiAgentWildfireEnvV3
from wildfire_rl.routing import compute_step_action, get_frontier_target, get_nearest_fire_target


class HybridMultiAgentWildfireEnv(MultiAgentWildfireEnvV3):
    """Hybrid environment separating high-level coordination and low-level routing.

    Action Space:
        MultiDiscrete([5] * num_agents):
            - 0: Target North-West quadrant sector
            - 1: Target North-East quadrant sector
            - 2: Target South-West quadrant sector
            - 3: Target South-East quadrant sector
            - 4: Target Global Nearest active fire

    Low-level Routing:
        - NearestFire routing (default) or Frontier routing.
        - The planner computes step actions automatically to route agents to target coordinates.
    """

    def __init__(
        self,
        state_tensor: np.ndarray | None = None,
        tensor_path: str | Path | None = None,
        config: EnvConfig | None = None,
        num_agents: int = 3,
        routing_strategy: str = "nearest_fire",
    ) -> None:
        super().__init__(
            state_tensor=state_tensor,
            tensor_path=tensor_path,
            config=config,
            num_agents=num_agents,
        )
        self.routing_strategy = routing_strategy.lower()

        # Action space is high-level sector/target selection (0 to 4) for each agent
        self.action_space = spaces.MultiDiscrete([5] * self.num_agents)
        self._target_coordinates: list[tuple[int, int]] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        # Check for Phase 5 robustness testing options
        if options and options.get("robustness_testing", False):
            from wildfire_rl.envs.randomized_fire_configs import randomize_env_robustness

            randomize_env_robustness(self, self.np_random)
            obs = self._obs()
            self._initial_fire_total = float(self.state[0].sum()) or 1.0

        self._target_coordinates = [tuple(pos) for pos in self.agent_positions]
        return obs, info

    def step(self, actions) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        actions = np.atleast_1d(actions)

        # 1. Low-level Routing: convert high-level sector targets to navigation step actions
        step_actions = []
        target_coords = []
        fire_channel = self.state[0]

        for i in range(self.num_agents):
            ax, ay = self.agent_positions[i]
            sector_idx = int(actions[i])

            # Resolve coordinates using the active routing strategy
            if self.routing_strategy == "frontier":
                tx, ty = get_frontier_target(
                    fire_channel, ax, ay, sector_idx, self.cfg.spread_threshold
                )
            else:
                tx, ty = get_nearest_fire_target(
                    fire_channel, ax, ay, sector_idx, self.cfg.spread_threshold
                )

            target_coords.append((tx, ty))

            # Compute the low-level step action (0=up, 1=down, 2=left, 3=right, 4=stay)
            step_action = compute_step_action(ax, ay, tx, ty, self.grid_size)
            step_actions.append(step_action)

        self._target_coordinates = target_coords

        # 2. Execute movement step actions
        for i in range(self.num_agents):
            self._move(i, step_actions[i])

        # Record visited cells BEFORE suppression (measures exploration)
        step_positions = [tuple(p) for p in self.agent_positions]
        for cell in step_positions:
            self._visited_cells.add(cell)

        # 3. Apply suppression and environment dynamics (movement already executed above).
        # We temporarily stub _move in step execution since we handled movement ourselves
        original_move = self._move
        self._move = lambda idx, act: None  # No-op since movement is already executed

        try:
            obs, reward, terminated, truncated, info = super().step(step_actions)
        finally:
            self._move = original_move

        # Append path-planning diagnostics for visual reporting
        info["target_coordinates"] = target_coords
        info["step_actions"] = step_actions

        return obs, reward, terminated, truncated, info
