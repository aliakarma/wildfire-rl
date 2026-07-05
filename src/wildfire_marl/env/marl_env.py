"""PettingZoo ParallelEnv wrapper for the wildfire multi-agent environment.

Implements N cooperating firefighting agents moving and treating cells over the
Cell2Fire simulation. Each agent observes an egocentric local crop plus global context.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from wildfire_marl.env.single_agent_env import FireSuppressionEnv
from wildfire_marl.env.rewards import InfrastructureWeightedReward, Reward
from wildfire_marl.eval.metrics import coordination_efficiency


class MultiAgentFireEnv(ParallelEnv):
    """N cooperating firefighting agents operating on top of a single-agent env."""

    metadata = {"render_modes": ["rgb_array"], "name": "multi_agent_fire_v0"}

    def __init__(
        self,
        num_agents: int = 3,
        crop_size: int = 9,
        coordination_penalty: float = 0.1,
        fire_map: str = "Sub40x40",
        **kwargs: Any,
    ):
        """
        Args:
            num_agents: Number of cooperating agents.
            crop_size: Size of the egocentric view square (default 9x9).
            coordination_penalty: Reward penalty for redundant treatments.
            fire_map: Instance map name.
            **kwargs: Passthrough arguments to FireSuppressionEnv.
        """
        super().__init__()
        self._num_agents = int(num_agents)
        self.crop_size = int(crop_size)
        if self.crop_size % 2 == 0:
            raise ValueError("crop_size must be an odd integer")
        self.crop_radius = self.crop_size // 2
        self.coordination_penalty = float(coordination_penalty)

        # Underlying single-agent env to manage simulator state
        self.env = FireSuppressionEnv(fire_map=fire_map, **kwargs)
        self.height, self.width = self.env.height, self.env.width

        # PettingZoo agent IDs
        self.agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.possible_agents = self.agents[:]

        # Action space: Discrete(6)
        # 0: Stay, 1: Move Up, 2: Move Down, 3: Move Left, 4: Move Right, 5: Treat current cell
        self.action_spaces = {agent: spaces.Discrete(6) for agent in self.agents}

        # Observation space: (8, crop_size, crop_size)
        self.observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(8, self.crop_size, self.crop_size),
                dtype=np.float32,
            )
            for agent in self.agents
        }

        self.agent_positions: dict[str, tuple[int, int]] = {}
        self.agent_step_positions: list[list[tuple[int, int]]] = []

    def observation_space(self, agent: str) -> spaces.Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self.action_spaces[agent]

    def _get_obs(self, agent: str) -> np.ndarray:
        """Construct the 8-channel egocentric crop observation for an agent."""
        y, x = self.agent_positions[agent]
        R = self.crop_radius

        # Channel sources
        fire_grid = (self.env.fire_state > 0).astype(np.float32)
        treated_grid = (self.env.fire_state < 0).astype(np.float32)
        fuel_grid = self.env.fuel_mask.astype(np.float32)
        crit_grid = self.env.criticality.astype(np.float32) if self.env.criticality is not None else np.zeros((self.height, self.width), dtype=np.float32)

        # Occupancy of other agents
        occ_grid = np.zeros((self.height, self.width), dtype=np.float32)
        for other, pos in self.agent_positions.items():
            if other != agent:
                occ_grid[pos[0], pos[1]] = 1.0

        # Global features
        num_burning = np.sum(fire_grid)
        global_fire_density = float(num_burning) / float(self.env.num_cells)

        # Target features (or fallback to self coordinates if target matches self position)
        if hasattr(self, "strategic_targets") and agent in self.strategic_targets:
            ty, tx = self.strategic_targets[agent]
            rel_ty = float(ty - y) / float(self.height)
            rel_tx = float(tx - x) / float(self.width)
        else:
            rel_ty = float(y) / float(self.height - 1) if self.height > 1 else 0.0
            rel_tx = float(x) / float(self.width - 1) if self.width > 1 else 0.0

        # Construct crops using zero padding
        obs = np.zeros((8, self.crop_size, self.crop_size), dtype=np.float32)
        grids = [fire_grid, treated_grid, fuel_grid, crit_grid, occ_grid]

        for c_idx, grid in enumerate(grids):
            padded = np.pad(grid, R, mode="constant", constant_values=0.0)
            obs[c_idx] = padded[y : y + self.crop_size, x : x + self.crop_size]

        # Homogeneous global channels
        obs[5] = global_fire_density
        obs[6] = rel_ty
        obs[7] = rel_tx

        return obs

    def _get_info(self, agent: str) -> dict[str, Any]:
        info = self.env._info()
        y, x = self.agent_positions[agent]
        info["agent_position"] = (y, x)
        info["action_mask"] = self.action_masks(agent)
        return info

    def action_masks(self, agent: str) -> np.ndarray:
        """True where action is valid. Treatment is only valid on unburned fuel cell."""
        y, x = self.agent_positions[agent]
        is_fuel = self.env.fuel_mask[y, x] > 0
        is_clean = self.env.fire_state[y, x] == 0
        treat_valid = is_fuel and is_clean

        # Movement is always valid (boundary hits just keep agent in place)
        return np.array([True, True, True, True, True, treat_valid], dtype=bool)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        options = options or {}
        # Reset inner env. Simulator RNG seed gets sampled here.
        _, env_info = self.env.reset(seed=seed, options=options)

        self.agents = self.possible_agents[:]

        # Initialize agent positions around center
        center_y, center_x = self.height // 2, self.width // 2
        self.agent_positions = {}
        self.strategic_targets = {}
        for i, agent in enumerate(self.agents):
            # Cluster around center
            offset_y = (i // 3) - 1
            offset_x = (i % 3) - 1
            y = int(np.clip(center_y + offset_y * 2, 0, self.height - 1))
            x = int(np.clip(center_x + offset_x * 2, 0, self.width - 1))
            self.agent_positions[agent] = (y, x)
            self.strategic_targets[agent] = (y, x)

        self.agent_step_positions = [list(self.agent_positions.values())]

        obs_dict = {agent: self._get_obs(agent) for agent in self.agents}
        info_dict = {agent: self._get_info(agent) for agent in self.agents}

        return obs_dict, info_dict

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        # Process movements and queue treatments
        treatment_cells = []
        redundant_treatments = 0

        # We first track which cells are treated *this step* by each agent
        attempted_treatments = {}

        for agent in self.agents:
            if agent not in actions:
                continue
            act = int(actions[agent])
            y, x = self.agent_positions[agent]

            if act == 1:  # Up
                y = max(0, y - 1)
            elif act == 2:  # Down
                y = min(self.height - 1, y + 1)
            elif act == 3:  # Left
                x = max(0, x - 1)
            elif act == 4:  # Right
                x = min(self.width - 1, x + 1)

            # Update position
            self.agent_positions[agent] = (y, x)

            if act == 5:  # Treat
                # Check if already treated in previous steps
                already_treated = self.env.fire_state[y, x] < 0
                is_fuel = self.env.fuel_mask[y, x] > 0
                is_clean = self.env.fire_state[y, x] == 0

                if is_fuel and is_clean and not already_treated:
                    attempted_treatments[agent] = (y, x)
                else:
                    redundant_treatments += 1

        self.agent_step_positions.append(list(self.agent_positions.values()))

        # Resolve treatment conflicts
        unique_cells = set()
        for agent, cell in attempted_treatments.items():
            if cell in unique_cells:
                # Multiple agents treating the same cell in this step
                redundant_treatments += 1
            else:
                unique_cells.add(cell)

        # Apply treatments to binding
        patch = [y * self.width + x for y, x in unique_cells]
        if patch:
            self.env.binding.apply_actions(patch)
            self.env.prev_actions.update(patch)
        else:
            self.env.binding.apply_actions(None)

        # Progress simulator
        csvs = self.env.binding.progress_to_next_state()
        self.env._refresh_state(csvs)

        # Post-spread wrapper cascade logic
        if self.env.cascade_prob > 0.0 and self.env.asset_type is not None:
            from wildfire_marl.infra.cascade import cascade_step
            self.env.fire_state, newly_det, newly_ign = cascade_step(
                fire_state=self.env.fire_state,
                asset_type=self.env.asset_type,
                blast_radius=self.env.blast_radius,
                fuel_mask=self.env.fuel_mask,
                previously_detonated=self.env.previously_detonated,
                cascade_prob=self.env.cascade_prob,
                rng=self.env.np_random,
            )
            self.env.previously_detonated.update(newly_det)
            self.env.cascade_ignited_count += newly_ign

        # Base team reward
        base_reward = float(self.env.reward_func(action=patch))
        coordination_loss = redundant_treatments * self.coordination_penalty
        
        # Target compliance reward shaping (for training guidance)
        compliance_loss = 0.0
        if getattr(self, "apply_target_compliance", False) and hasattr(self, "strategic_targets"):
            for agent in self.agents:
                if agent in self.strategic_targets:
                    ay, ax = self.agent_positions[agent]
                    ty, tx = self.strategic_targets[agent]
                    compliance_loss += 0.05 * (abs(ty - ay) + abs(tx - ax))
                    
        shared_reward = base_reward - coordination_loss - compliance_loss

        self.env.iter += 1
        terminated = self.env.binding.finished
        truncated = (not terminated) and self.env.iter >= self.env.max_steps

        # Construct outputs
        obs_dict = {agent: self._get_obs(agent) for agent in self.agents}
        reward_dict = {agent: shared_reward for agent in self.agents}
        terminated_dict = {agent: terminated for agent in self.agents}
        truncated_dict = {agent: truncated for agent in self.agents}
        info_dict = {agent: self._get_info(agent) for agent in self.agents}

        # Calculate final metrics if done
        if terminated or truncated:
            ce = coordination_efficiency(self.agent_step_positions)
            for agent in self.agents:
                info_dict[agent]["coordination_efficiency"] = ce

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def get_global_state(self) -> np.ndarray:
        """Construct the 5-channel centralized global state."""
        state = np.zeros((5, self.height, self.width), dtype=np.float32)
        state[0] = (self.env.fire_state > 0).astype(np.float32)
        state[1] = (self.env.fire_state < 0).astype(np.float32)
        state[2] = self.env.fuel_mask.astype(np.float32)
        if self.env.criticality is not None:
            state[3] = self.env.criticality.astype(np.float32)
        for pos in self.agent_positions.values():
            state[4, pos[0], pos[1]] = 1.0
        return state

    def render(self) -> np.ndarray:
        im = self.env.render()
        # Overlay agent positions as white dots
        for agent, (y, x) in self.agent_positions.items():
            im[y, x] = (255, 255, 255)
        return im

    def close(self):
        self.env.close()
