"""Adaptive coordination environment wrapping PPO target decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from wildfire_rl.config import EnvConfig
from wildfire_rl.envs.dynamic_fire_scenarios import DynamicMultiAgentWildfireEnv
from wildfire_rl.routing import compute_step_action, get_frontier_target, get_nearest_fire_target


class AdaptiveHybridMultiAgentWildfireEnv(DynamicMultiAgentWildfireEnv):
    """Adaptive coordination multi-agent environment.

    Extends DynamicMultiAgentWildfireEnv with an expanded action space representing
    8 strategic coordination decisions.

    Action Space:
        MultiDiscrete([8] * num_agents):
            - 0: HOLD_SECTOR NW (defend NW quadrant)
            - 1: HOLD_SECTOR NE (defend NE quadrant)
            - 2: HOLD_SECTOR SW (defend SW quadrant)
            - 3: HOLD_SECTOR SE (defend SE quadrant)
            - 4: REQUEST_SUPPORT (assist the sector with the highest fire intensity)
            - 5: EMERGENCY_REASSIGN (reassign to the highest fire sector excluding current quadrant)
            - 6: GLOBAL_RESPONSE (target the closest active fire globally)
            - 7: FRONTIER_INTERCEPT (target the closest frontier cell globally)
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
            routing_strategy=routing_strategy,
        )
        # Expand PPO action space to 8 discrete actions
        self.action_space = spaces.MultiDiscrete([8] * self.num_agents)

    def _get_sector_loads(self) -> np.ndarray:
        """Return the sum of fire intensity in each of the 4 quadrant sectors."""
        fire = self.state[0]
        half = self.grid_size // 2

        nw = float(fire[0:half, 0:half].sum())
        ne = float(fire[0:half, half : self.grid_size].sum())
        sw = float(fire[half : self.grid_size, 0:half].sum())
        se = float(fire[half : self.grid_size, half : self.grid_size].sum())

        return np.array([nw, ne, sw, se])

    def _get_agent_quadrant(self, ax: int, ay: int) -> int:
        """Return the quadrant index (0-3) of the agent's current position."""
        half = self.grid_size // 2
        if ax < half:
            return 0 if ay < half else 1
        else:
            return 2 if ay < half else 3

    def step(self, actions) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        actions = np.atleast_1d(actions)

        # Pre-calculate sector loads
        sector_loads = self._get_sector_loads()
        max_sector = int(np.argmax(sector_loads)) if sector_loads.sum() > 0.01 else 4

        step_actions = []
        target_coords = []
        fire_channel = self.state[0]

        for i in range(self.num_agents):
            ax, ay = self.agent_positions[i]
            ppo_action = int(actions[i])
            resolved_sector = 4

            # Map the 8 discrete actions to resolved sectors or global targeting
            if ppo_action in [0, 1, 2, 3]:
                # NW, NE, SW, SE HOLD_SECTOR
                resolved_sector = ppo_action
            elif ppo_action == 4:
                # REQUEST_SUPPORT: assist the sector with max fire
                resolved_sector = max_sector
            elif ppo_action == 5:
                # EMERGENCY_REASSIGN: assist max fire sector excluding current agent quadrant
                agent_quad = self._get_agent_quadrant(ax, ay)
                masked_loads = sector_loads.copy()
                masked_loads[agent_quad] = -1.0  # exclude current quadrant
                if masked_loads.max() > 0.01:
                    resolved_sector = int(np.argmax(masked_loads))
                else:
                    resolved_sector = max_sector
            elif ppo_action == 6:
                # GLOBAL_RESPONSE
                resolved_sector = 4
            elif ppo_action == 7:
                # FRONTIER_INTERCEPT
                resolved_sector = 4

            # 2. Path routing target coordinate calculations
            if ppo_action == 7:
                # Target closest active frontier cell globally
                tx, ty = get_frontier_target(fire_channel, ax, ay, 4, self.cfg.spread_threshold)
            else:
                # Target closest active fire cell inside the resolved sector
                tx, ty = get_nearest_fire_target(
                    fire_channel, ax, ay, resolved_sector, self.cfg.spread_threshold
                )

            target_coords.append((tx, ty))

            # Compute low-level step movement (0-4) towards target coordinates
            step_action = compute_step_action(ax, ay, tx, ty, self.grid_size)
            step_actions.append(step_action)

        self._target_coordinates = target_coords

        # Execute movement step actions
        for i in range(self.num_agents):
            self._move(i, step_actions[i])

        # Record visited cells BEFORE suppression
        step_positions = [tuple(p) for p in self.agent_positions]
        for cell in step_positions:
            self._visited_cells.add(cell)

        # Apply suppression and environment dynamic events
        original_move = self._move
        self._move = lambda idx, act: None  # No-op during base step

        try:
            # We call step on DynamicMultiAgentWildfireEnv which triggers wind/ignitions
            obs, reward, terminated, truncated, info = super().step(step_actions)
        finally:
            self._move = original_move

        # Append path-planning diagnostics for visual reporting
        info["target_coordinates"] = target_coords
        info["step_actions"] = step_actions

        return obs, reward, terminated, truncated, info
