import pytest
import numpy as np
from wildfire_rl.config import EnvConfig
from wildfire_rl.envs.multi_agent_v3 import MultiAgentWildfireEnvV3


def test_unique_starting_positions():
    """Verify that starting positions do not overlap for any team size up to 10."""
    tensor = np.zeros((7, 32, 32), dtype=np.float32)
    tensor[0, 16, 16] = 1.0  # mock fire center

    for n_agents in range(1, 11):
        env = MultiAgentWildfireEnvV3(state_tensor=tensor, num_agents=n_agents)
        env.reset(seed=0)
        positions = [tuple(p) for p in env.agent_positions]
        unique_positions = set(positions)
        assert len(unique_positions) == n_agents, f"Overlapping positions detected for {n_agents} agents: {positions}"


def test_v3_reward_active_suppression():
    """Verify V3 reward components measure agent-caused suppression and exclude decay leakage."""
    tensor = np.zeros((7, 32, 32), dtype=np.float32)
    # Put fire right under the starting position of Agent 0 (16, 16)
    tensor[0, 16, 16] = 1.0

    cfg = EnvConfig()
    cfg.reward_v3.enabled = True
    cfg.reward_v3.spread_reduction_weight = 5.0
    cfg.reward_v3.overlap_penalty = -0.05
    cfg.reward_v3.suppression_threshold = 0.05

    env = MultiAgentWildfireEnvV3(state_tensor=tensor, config=cfg, num_agents=1)
    env.reset(seed=0)

    # Step 1: Stay at (16, 16) and suppress the fire
    obs, reward, term, trunc, info = env.step([4])  # 4 = stay
    comps = info.get("reward_components", {})

    # Agent should have caused suppression
    assert comps["spread_reduction"] > 0, "Agent-caused suppression should be positive"
    # Containment stability should increment because agent_suppressed > threshold (0.05)
    assert comps["containment_stability"] > 0, "Containment streak should increment"

    # Step 2: Teleport agent far away to (0, 0) and take a step
    env.agent_positions[0] = [0, 0]
    obs, reward, term, trunc, info = env.step([4])  # stay at (0, 0)
    comps_step2 = info.get("reward_components", {})

    # No suppression should occur here
    assert comps_step2["spread_reduction"] == 0, "No agent-caused suppression should occur"
    # Containment streak should reset to 0
    assert comps_step2["containment_stability"] == 0, "Containment streak should reset to 0"
