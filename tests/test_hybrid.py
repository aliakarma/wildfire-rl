import numpy as np

from wildfire_rl.config import EnvConfig
from wildfire_rl.envs.hybrid_multi_agent import HybridMultiAgentWildfireEnv
from wildfire_rl.routing import compute_step_action, get_nearest_fire_target


def test_routing_logic():
    """Verify that Manhattan step routing calculates correct movement actions."""
    # Stay at target
    assert compute_step_action(10, 10, 10, 10) == 4

    # Move up (row index decreases)
    assert compute_step_action(10, 10, 5, 10) == 0

    # Move down (row index increases)
    assert compute_step_action(10, 10, 15, 10) == 1

    # Move left (col index decreases)
    assert compute_step_action(10, 10, 10, 5) == 2

    # Move right (col index increases)
    assert compute_step_action(10, 10, 10, 15) == 3


def test_sector_nearest_fire_router():
    """Verify nearest fire router targets the correct sector quadrant."""
    # Grid size 32x32, center is (16, 16)
    fire_channel = np.zeros((32, 32), dtype=np.float32)
    # Fire in Sector 0 (NW): row 5, col 5
    fire_channel[5, 5] = 1.0
    # Fire in Sector 1 (NE): row 5, col 25
    fire_channel[5, 25] = 1.0

    # Agent at center (16, 16) targeting Sector 0 (NW)
    tx, ty = get_nearest_fire_target(fire_channel, 16, 16, 0, spread_threshold=0.2)
    assert tx == 5 and ty == 5, f"Expected NW target (5, 5), got ({tx}, {ty})"

    # Agent at center (16, 16) targeting Sector 1 (NE)
    tx, ty = get_nearest_fire_target(fire_channel, 16, 16, 1, spread_threshold=0.2)
    assert tx == 5 and ty == 25, f"Expected NE target (5, 25), got ({tx}, {ty})"


def test_hybrid_multi_agent_step():
    """Verify HybridMultiAgentEnv step maps sector selections to path-routing movements."""
    tensor = np.zeros((7, 32, 32), dtype=np.float32)
    # Fire in Sector 0 (NW) at (5, 5)
    tensor[0, 5, 5] = 1.0

    cfg = EnvConfig()
    cfg.reward_v3.enabled = True
    cfg.reward_v3.spread_reduction_weight = 5.0
    cfg.reward_v3.overlap_penalty = -0.05
    cfg.reward_v3.suppression_threshold = 0.05

    env = HybridMultiAgentWildfireEnv(
        state_tensor=tensor, config=cfg, num_agents=1, routing_strategy="nearest_fire"
    )
    obs, _ = env.reset(seed=0)

    # Initial position is [16, 16]
    assert env.agent_positions[0] == [16, 16]

    # Target Sector 0 (NW), which should resolve to fire at (5, 5)
    # Step action should move agent UP (row 16 -> 15)
    obs, reward, term, trunc, info = env.step([0])

    assert env.agent_positions[0] == [
        15,
        16,
    ], f"Expected agent to move UP to [15, 16], got {env.agent_positions[0]}"
    assert info["target_coordinates"][0] == (
        5,
        5,
    ), f"Expected target coordinates to be (5, 5), got {info['target_coordinates'][0]}"
    assert (
        info["step_actions"][0] == 0
    ), f"Expected step action to be 0 (UP), got {info['step_actions'][0]}"
