"""Dynamic multi-agent wildfire environment with dynamic fire scenarios.

Subclass of :class:`HybridMultiAgentWildfireEnv` that injects:
    1. **Moving Wind Fronts**: Wind angles shift dynamically over steps.
    2. **Asymmetric NW Sector Overload**: Fire spreads faster in the NW quadrant.
    3. **NE Sector Collapse Events**: Multiple ignitions trigger simultaneously in NE.
    4. **Multi-Wave Ignitions**: New fires spawn at step thresholds.
    5. **Dynamic Fuel Corridor**: High-risk drought fuel corridor appears mid-episode.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wildfire_rl.config import EnvConfig
from wildfire_rl.envs.hybrid_multi_agent import HybridMultiAgentWildfireEnv


class DynamicMultiAgentWildfireEnv(HybridMultiAgentWildfireEnv):
    """Hybrid environment with dynamic, asymmetric, and unpredictable fire scenarios."""

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
        # Store initial fuel map for dynamic corridor resets
        self._initial_fuel_map = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self._initial_fuel_map = self.state[1].copy()
        return obs, info

    def step(self, actions) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # 1. Inject dynamic scenarios based on current step
        # Retrieve config flags (defaults if not in config)
        dynamic_cfg = getattr(self.cfg, "dynamic_scenarios", {})
        
        # A. Moving Wind Fronts: shift wind vectors every 40 steps
        if dynamic_cfg.get("moving_wind", True) and self.current_step > 0 and self.current_step % 40 == 0:
            angle = self.np_random.uniform(0, 2 * np.pi)
            magnitude = self.np_random.uniform(0.04, 0.18)
            self.state[2] = np.cos(angle) * magnitude
            self.state[3] = np.sin(angle) * magnitude

        # B. NE Sector Collapse: Northeast quadrant NE gets overloaded with ignitions at step 30
        if dynamic_cfg.get("sector_collapse", True) and self.current_step == 30:
            # Northeast quadrant: row in [2, 14], col in [18, 30]
            for _ in range(3):
                rx = self.np_random.integers(2, 14)
                ry = self.np_random.integers(18, 30)
                self.state[0, rx, ry] = 1.0

        # C. Multi-Wave Ignitions: new fires spawn at steps 50 and 100
        if dynamic_cfg.get("multi_wave_ignitions", True) and self.current_step in [50, 100]:
            for _ in range(2):
                rx = self.np_random.integers(2, self.grid_size - 2)
                ry = self.np_random.integers(2, self.grid_size - 2)
                self.state[0, rx, ry] = 1.0

        # D. Dynamic Fuel Corridor: dry vegetation horizontally across rows 15-18 at step 60
        if dynamic_cfg.get("fuel_corridor", True) and self.current_step == 60:
            self.state[1, 15:19, :] = 1.0  # max fuel density

        # E. NW Asymmetric Overload: Northwest quadrant has higher spread rate
        # We simulate this by checking if fire is in NW (row < 16, col < 16) and expanding it slightly
        if dynamic_cfg.get("asymmetric_overload", True) and self.current_step > 0:
            nw_fire = self.state[0, 0:16, 0:16]
            # Probabilistically accelerate NW fire spreading
            active = nw_fire > self.cfg.spread_threshold
            if active.any() and self.np_random.uniform(0, 1) < 0.15:
                # Add tiny fire increment to neighbors of active cells
                for x in range(1, 15):
                    for y in range(1, 15):
                        if nw_fire[x, y] > self.cfg.spread_threshold:
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                if self.state[0, x+dx, y+dy] < 0.1:
                                    self.state[0, x+dx, y+dy] += 0.12

        # 2. Run standard hybrid navigation step
        obs, reward, terminated, truncated, info = super().step(actions)
        return obs, reward, terminated, truncated, info
