"""Unit and integration tests for Phase 3 critical infrastructure features."""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_marl.infra.build_infrastructure import build_infrastructure
from wildfire_marl.infra.cascade import cascade_step
from wildfire_marl.env.rewards import InfrastructureWeightedReward, FireSizeReward


def test_build_infrastructure_shapes_and_values():
    """Verify built infrastructure rasters have correct shapes, bounds, and order."""
    for region in ["saudi", "california"]:
        asset_type, criticality, blast_radius = build_infrastructure(region, grid=32)
        
        # Check shapes
        assert asset_type.shape == (32, 32)
        assert criticality.shape == (32, 32)
        assert blast_radius.shape == (32, 32)
        
        # Check types
        assert asset_type.dtype == np.int32
        assert criticality.dtype == np.float32
        assert blast_radius.dtype == np.int32
        
        # Check criticality range [0, 1]
        assert 0.0 <= criticality.min() <= criticality.max() <= 1.0
        
        # Verify that assets have correct default values and blast radii
        # Refinery = 1, Pipeline = 2, Storage = 3, Industrial = 4
        # Default blast: Refineries/Storage = 3, Pipelines/Industrial = 2
        for r in range(32):
            for c in range(32):
                code = asset_type[r, c]
                if code > 0:
                    assert criticality[r, c] > 0.0
                    if code in (1, 3):
                        assert blast_radius[r, c] == 3
                    elif code in (2, 4):
                        assert blast_radius[r, c] == 2


def test_cascade_detonation_logic():
    """Verify post-spread wrapper-level cascade logic behaves correctly."""
    grid = 8
    fire_state = np.zeros((grid, grid), dtype=np.int8)
    asset_type = np.zeros((grid, grid), dtype=np.int32)
    blast_radius = np.zeros((grid, grid), dtype=np.int32)
    fuel_mask = np.ones((grid, grid), dtype=np.float32)
    previously_detonated = set()
    rng = np.random.default_rng(42)

    # Place a Refinery at (3, 3) with blast radius 2
    asset_type[3, 3] = 1
    blast_radius[3, 3] = 2

    # Step 1: No fire on asset -> no cascade
    new_fire, newly_det, newly_ign = cascade_step(
        fire_state=fire_state,
        asset_type=asset_type,
        blast_radius=blast_radius,
        fuel_mask=fuel_mask,
        previously_detonated=previously_detonated,
        cascade_prob=1.0,  # deterministic ignitions
        rng=rng,
    )
    assert len(newly_det) == 0
    assert newly_ign == 0
    assert np.array_equal(new_fire, fire_state)

    # Step 2: Ignite asset -> cascades to radius 2
    fire_state[3, 3] = 1  # Ignite
    new_fire, newly_det, newly_ign = cascade_step(
        fire_state=fire_state,
        asset_type=asset_type,
        blast_radius=blast_radius,
        fuel_mask=fuel_mask,
        previously_detonated=previously_detonated,
        cascade_prob=1.0,
        rng=rng,
    )
    assert len(newly_det) == 1
    assert (3 * grid + 3) in newly_det
    
    # Radius = 2: all cells within circular/Euclidean distance 2 of (3,3) should ignite
    # dx^2 + dy^2 <= 4
    # Check that (3, 3) itself is unaffected (already ignited), but (3, 1), (3, 2), (3, 4), (3, 5), (1, 3), (2, 3), (4, 3), (5, 3), and diagonals (2, 2), (2, 4), (4, 2), (4, 4) are ignited
    expected_ignitions = 0
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if (dx == 0 and dy == 0) or (dx * dx + dy * dy > 4):
                continue
            assert new_fire[3 + dx, 3 + dy] == 1.0
            expected_ignitions += 1
    assert newly_ign == expected_ignitions

    # Update previously detonated
    previously_detonated.update(newly_det)

    # Step 3: Already detonated should not detonate again
    new_fire_2, newly_det_2, newly_ign_2 = cascade_step(
        fire_state=new_fire,
        asset_type=asset_type,
        blast_radius=blast_radius,
        fuel_mask=fuel_mask,
        previously_detonated=previously_detonated,
        cascade_prob=1.0,
        rng=rng,
    )
    assert len(newly_det_2) == 0
    assert newly_ign_2 == 0


class DummyEnv:
    """Mock environment for testing InfrastructureWeightedReward."""
    def __init__(self, fire_state, asset_type, catastrophe_weight):
        self.fire_state = fire_state
        self.num_cells = fire_state.size
        self.asset_type = asset_type
        self.catastrophe_weight = catastrophe_weight
        self.asset_values = {1: 10.0, 2: 4.0, 3: 6.0, 4: 3.0}


def test_infrastructure_weighted_reward():
    """Verify that InfrastructureWeightedReward orders refinery > pipeline > empty."""
    grid = 8
    fire_state = np.zeros((grid, grid), dtype=np.int8)
    asset_type = np.zeros((grid, grid), dtype=np.int32)
    
    # 1. Setup mock env with a Refinery (1) and a Pipeline (2)
    asset_type[1, 1] = 1  # Refinery
    asset_type[2, 2] = 2  # Pipeline

    env = DummyEnv(fire_state, asset_type, catastrophe_weight=2.0)
    reward_func = InfrastructureWeightedReward(env, scale_fire=10.0)

    # Case A: No fire -> reward = 0
    assert reward_func() == 0.0

    # Case B: Fire on non-infrastructure cell (0, 0)
    fire_state[0, 0] = 1
    r_empty = reward_func()
    # fire penalty: -1/64 * 10 = -0.15625
    assert np.isclose(r_empty, -0.15625)

    # Case C: Fire on Pipeline cell (2, 2) instead
    fire_state.fill(0)
    fire_state[2, 2] = 1
    r_pipeline = reward_func()
    # fire penalty: -1/64 * 10 = -0.15625
    # pipeline penalty: -2.0 (weight) * 4.0 (pipeline val) * 1 (burning) = -8.0
    # total: -8.15625
    assert np.isclose(r_pipeline, -8.15625)

    # Case D: Fire on Refinery cell (1, 1) instead
    fire_state.fill(0)
    fire_state[1, 1] = 1
    r_refinery = reward_func()
    # fire penalty: -0.15625
    # refinery penalty: -2.0 * 10.0 * 1 = -20.0
    # total: -20.15625
    assert np.isclose(r_refinery, -20.15625)

    # Verify order of penalties: Refinery > Pipeline > Empty
    assert r_refinery < r_pipeline < r_empty
