"""Unit tests verifying the PettingZoo multi-agent environment wrapper and coordination metrics.
"""

from __future__ import annotations

import pytest
import numpy as np

from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.eval.metrics import coordination_efficiency


def test_marl_env_basic():
    # Setup test env (on small Sub40x40 map)
    env = MultiAgentFireEnv(
        num_agents=3,
        crop_size=9,
        coordination_penalty=0.1,
        fire_map="Sub40x40",
    )

    assert env.num_agents == 3
    assert len(env.agents) == 3
    assert "agent_0" in env.agents
    assert "agent_1" in env.agents
    assert "agent_2" in env.agents

    obs_dict, info_dict = env.reset(seed=42)

    assert isinstance(obs_dict, dict)
    assert isinstance(info_dict, dict)
    assert len(obs_dict) == 3
    assert len(info_dict) == 3

    for agent in env.agents:
        obs = obs_dict[agent]
        assert obs.shape == (8, 9, 9)
        assert isinstance(info_dict[agent]["action_mask"], np.ndarray)
        assert len(info_dict[agent]["action_mask"]) == 6

    # Test step execution
    # Issue a treat action (5) or movement (0-4)
    actions = {
        "agent_0": 1,  # Up
        "agent_1": 5,  # Treat
        "agent_2": 0,  # Stay
    }

    obs_dict, rewards, terminated, truncated, infos = env.step(actions)

    assert isinstance(obs_dict, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminated, dict)
    assert isinstance(truncated, dict)
    assert isinstance(infos, dict)

    assert len(rewards) == 3
    assert len(terminated) == 3

    env.close()


def test_coordination_efficiency_metric():
    # If agents occupy separate cells: CE is 1.0
    steps_separated = [
        [(0, 0), (1, 1), (2, 2)],
        [(0, 1), (1, 2), (2, 3)],
    ]
    assert coordination_efficiency(steps_separated) == 1.0

    # If agents occupy same cells: CE is less than 1.0
    steps_overlap = [
        [(0, 0), (0, 0), (2, 2)],  # 2 agents on same cell (0, 0)
        [(1, 1), (1, 1), (1, 1)],  # all 3 on same cell
    ]
    # step 1: total=3, distinct=2
    # step 2: total=3, distinct=1
    # sum total = 6, sum distinct = 3
    # CE = 3 / 6 = 0.5
    assert coordination_efficiency(steps_overlap) == 0.5


def test_division_of_labor_metrics():
    from wildfire_marl.eval.coordination import spatial_division_of_labor, redundant_treatment_rate

    # Perfect division of labor (paths completely disjoint)
    paths_disjoint = {
        "agent_0": [(0, 0), (0, 1), (0, 2)],
        "agent_1": [(5, 5), (5, 6), (5, 7)],
    }
    assert spatial_division_of_labor(paths_disjoint) == 1.0

    # No division of labor (paths identical)
    paths_identical = {
        "agent_0": [(0, 0), (0, 1), (0, 2)],
        "agent_1": [(0, 0), (0, 1), (0, 2)],
    }
    assert spatial_division_of_labor(paths_identical) == 0.0

    # Half overlap
    paths_half = {
        "agent_0": [(0, 0), (0, 1)],
        "agent_1": [(0, 1), (0, 2)],
    }
    # Union = {(0,0), (0,1), (0,2)} size 3. Intersection = {(0,1)} size 1.
    # Jaccard distance = 1 - 1/3 = 2/3 = 0.6666...
    assert abs(spatial_division_of_labor(paths_half) - (2.0 / 3.0)) < 1e-5

    # Redundant treatment rate
    assert redundant_treatment_rate(10, 2) == 0.2
    assert redundant_treatment_rate(0, 0) == 0.0


def test_marl_action_masking():
    env = MultiAgentFireEnv(
        num_agents=2,
        crop_size=9,
        coordination_penalty=0.1,
        fire_map="Sub40x40",
    )

    env.reset(seed=42)

    # Place agent_0 on a non-fuel coordinate (we know fuel mask has non-fuel cells,
    # let's set agent position to top-left corner which might be non-fuel or fuel,
    # but we can set fuel mask to 0 at the agent's position to force it).
    agent = "agent_0"
    y, x = env.agent_positions[agent]
    
    # Check mask validity
    mask = env.action_masks(agent)
    # The first 5 movement actions are always valid
    assert np.all(mask[:5] == True)

    # Force non-fuel cell
    env.env.fuel_mask[y, x] = 0.0
    mask_nonfuel = env.action_masks(agent)
    assert mask_nonfuel[5] == False

    # Force clean fuel cell
    env.env.fuel_mask[y, x] = 1.0
    env.env.fire_state[y, x] = 0
    mask_clean = env.action_masks(agent)
    assert mask_clean[5] == True

    # Force burning fuel cell
    env.env.fire_state[y, x] = 1
    mask_burning = env.action_masks(agent)
    assert mask_burning[5] == False

    env.close()
