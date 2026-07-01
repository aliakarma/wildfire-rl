"""Multi-agent wildfire environment V3 — coordination-focused reward shaping.

Subclass of :class:`MultiAgentWildfireEnv` that implements Phase 3 fixes:
    1. **Agent-caused suppression only**: Rewards are based on the difference
       in fire intensity immediately before and after the agent's suppression
       is applied, completely ignoring natural decay.
    2. **Containment leakage fix**: The streak bonus only increments if agents
       actively suppress fire (suppression delta > threshold), preventing
       passive streak accumulation from natural decay.
    3. **Balanced scales**: Massively reduced overlap penalty (default -0.05) and
       amplified exploration/suppression rewards to prevent deterministic collapse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import convolve

from wildfire_rl.config import EnvConfig
from wildfire_rl.envs import dynamics
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv

# Kernel for counting orthogonal neighbors (up/down/left/right).
_NEIGHBOR_KERNEL = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


class MultiAgentWildfireEnvV3(MultiAgentWildfireEnv):
    """Multi-agent env with V3 coordination-focused reward shaping."""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        # V3 tracking state
        self._visited_cells: set[tuple[int, int]] = set()
        self._containment_streak = 0
        self._reward_components: dict[str, float] = {}

        # Record initial positions as visited
        for pos in self.agent_positions:
            self._visited_cells.add((pos[0], pos[1]))

        return obs, info

    def step(self, actions) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        actions = np.atleast_1d(actions)

        # Move agents
        for i in range(self.num_agents):
            self._move(i, int(actions[i]))

        # Record visited cells BEFORE suppression (measures exploration)
        step_positions: list[tuple[int, int]] = []
        for pos in self.agent_positions:
            cell = (pos[0], pos[1])
            step_positions.append(cell)
            self._visited_cells.add(cell)

        # Capture fire state BEFORE suppression to measure agent-caused suppression
        fire_before = self.state[0].copy()

        # Apply suppression
        dynamics.apply_suppression(
            self.state, [tuple(p) for p in self.agent_positions], self.cfg
        )

        # Capture fire state AFTER suppression (but BEFORE spread/decay)
        fire_after_suppression = self.state[0].copy()

        # Pure agent-caused suppression (ignore decay/spread)
        agent_suppressed = float((fire_before - fire_after_suppression).sum())

        # Apply spread and decay dynamics
        self.state[0] = dynamics.spread_fire(self.state, self.cfg, self.np_random)
        dynamics.decay_and_deplete(self.state, self.cfg)

        # Compute current fire (for info/termination)
        curr_fire_total = float(self.state[0].sum())

        # ── V3 Reward Computation ──
        rv3 = self.cfg.reward_v3
        components: dict[str, float] = {}

        # 1. Spread Reduction: reward ONLY agent-caused suppression
        # Normalized by initial fire total to make it comparable
        components["spread_reduction"] = (
            rv3.spread_reduction_weight * agent_suppressed / self._initial_fire_total
        )

        # 2. Frontier Blocking: reward agents positioned at fire frontier cells
        fire_channel = self.state[0]
        active = fire_channel > self.cfg.spread_threshold
        if active.any():
            active_neighbors = convolve(
                active.astype(np.int32), _NEIGHBOR_KERNEL, mode="constant", cval=0
            )
            # Frontier = burning cell with < 4 burning neighbors
            frontier = active & (active_neighbors < 4)
            frontier_bonus = 0.0
            for pos in self.agent_positions:
                x, y = pos
                if frontier[x, y]:
                    frontier_bonus += rv3.frontier_blocking_weight
            components["frontier_blocking"] = frontier_bonus
        else:
            components["frontier_blocking"] = 0.0

        # 3. Coverage: unique cells visited / grid_area (exploration incentive)
        grid_area = self.grid_size * self.grid_size
        components["coverage"] = (
            rv3.coverage_weight * len(self._visited_cells) / grid_area
        )

        # 4. Overlap Penalty: penalize agents sharing the same cell (massively reduced)
        unique_positions = set(step_positions)
        n_overlapping_pairs = len(step_positions) - len(unique_positions)
        components["overlap_penalty"] = rv3.overlap_penalty * n_overlapping_pairs

        # 5. Containment Stability: streak bonus ONLY if agents actively suppress fire
        # Prevents decay leakage
        if agent_suppressed > rv3.suppression_threshold:
            self._containment_streak = min(
                self._containment_streak + 1, rv3.containment_streak_cap
            )
        else:
            self._containment_streak = 0

        components["containment_stability"] = (
            rv3.containment_stability_weight * self._containment_streak
        )

        # ── Aggregate Reward ──
        # Base normalized reward (same as V1 for baseline comparison)
        base_reward = float(-curr_fire_total / self._initial_fire_total)

        # Sum all V3 components
        v3_bonus = sum(components.values())
        reward = base_reward + v3_bonus

        # Store for diagnostics
        components["base_reward"] = base_reward
        components["total_reward"] = reward
        self._reward_components = components

        # Termination (same as V1)
        terminated = curr_fire_total < self.cfg.termination_fire_threshold
        truncated = self.current_step >= self.cfg.max_steps

        info = {
            "total_fire": curr_fire_total,
            "num_agents": self.num_agents,
            "reward_components": dict(components),
        }
        return self._obs(), reward, terminated, truncated, info
