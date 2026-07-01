"""Multi-agent wildfire environment V2 — enhanced reward shaping.

Subclass of :class:`MultiAgentWildfireEnv` that adds five reward components
designed to fix entropy collapse, encourage spatial coordination, and reward
strategic firefighting over blind local suppression.

Reward Components:
    1. **Spread Reduction**: Reward proportional to fire decrease since last step
    2. **Frontier Blocking**: Bonus for agents positioned at fire frontier cells
    3. **Coverage**: Reward for visiting unique cells (exploration incentive)
    4. **Overlap Penalty**: Penalize multiple agents at the same cell
    5. **Containment Stability**: Exponential bonus for consecutive fire-decreasing steps

All weights are configurable via :class:`RewardV2Config`.
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


class MultiAgentWildfireEnvV2(MultiAgentWildfireEnv):
    """Multi-agent env with V2 multi-component reward shaping.

    Drop-in replacement for ``MultiAgentWildfireEnv`` — same obs/action spaces,
    same dynamics, different reward signal.
    """

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        # V2 tracking state
        self._prev_fire_total = float(self.state[0].sum())
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

        # Apply dynamics (same as V1)
        dynamics.apply_suppression(
            self.state, [tuple(p) for p in self.agent_positions], self.cfg
        )
        self.state[0] = dynamics.spread_fire(self.state, self.cfg, self.np_random)
        dynamics.decay_and_deplete(self.state, self.cfg)

        # Compute current fire
        curr_fire_total = float(self.state[0].sum())

        # ── V2 Reward Computation ──
        rv2 = self.cfg.reward_v2
        components: dict[str, float] = {}

        # 1. Spread Reduction: reward = w * (prev - curr) / initial
        fire_delta = self._prev_fire_total - curr_fire_total
        components["spread_reduction"] = (
            rv2.spread_reduction_weight * fire_delta / self._initial_fire_total
        )

        # 2. Frontier Blocking: agents at fire frontier cells get bonus
        fire_channel = self.state[0]
        active = fire_channel > self.cfg.spread_threshold
        if active.any():
            active_neighbors = convolve(
                active.astype(np.int32), _NEIGHBOR_KERNEL, mode="constant", cval=0
            )
            # Frontier = burning cell with < 4 burning neighbors (edge of fire)
            frontier = active & (active_neighbors < 4)
            frontier_bonus = 0.0
            for pos in self.agent_positions:
                x, y = pos
                if frontier[x, y]:
                    frontier_bonus += rv2.frontier_blocking_weight
            components["frontier_blocking"] = frontier_bonus
        else:
            components["frontier_blocking"] = 0.0

        # 3. Coverage: unique cells visited this step / grid_area
        new_cells_this_step = sum(
            1 for c in step_positions if c not in self._visited_cells
            or self.current_step == 1  # count first step
        )
        grid_area = self.grid_size * self.grid_size
        components["coverage"] = (
            rv2.coverage_weight * len(self._visited_cells) / grid_area
        )

        # 4. Overlap Penalty: penalize agents sharing the same cell
        unique_positions = set(step_positions)
        n_overlapping_pairs = len(step_positions) - len(unique_positions)
        components["overlap_penalty"] = rv2.overlap_penalty * n_overlapping_pairs

        # 5. Containment Stability: streak bonus for consecutive fire-decrease steps
        if curr_fire_total < self._prev_fire_total:
            self._containment_streak = min(
                self._containment_streak + 1, rv2.containment_streak_cap
            )
        else:
            self._containment_streak = 0

        components["containment_stability"] = (
            rv2.containment_stability_weight * self._containment_streak
        )

        # ── Aggregate Reward ──
        # Base normalized reward (same as V1 for comparison baseline)
        base_reward = float(-curr_fire_total / self._initial_fire_total)

        # Sum all V2 components
        v2_bonus = sum(components.values())
        reward = base_reward + v2_bonus

        # Store for diagnostics
        components["base_reward"] = base_reward
        components["total_reward"] = reward
        self._reward_components = components

        # Update tracking
        self._prev_fire_total = curr_fire_total

        # Termination (same as V1)
        terminated = curr_fire_total < self.cfg.termination_fire_threshold
        truncated = self.current_step >= self.cfg.max_steps

        info = {
            "total_fire": curr_fire_total,
            "num_agents": self.num_agents,
            "reward_components": dict(components),
        }
        return self._obs(), reward, terminated, truncated, info
