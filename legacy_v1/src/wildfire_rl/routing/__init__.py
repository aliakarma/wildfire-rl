"""Routing module for Hybrid Path-Planning + RL Coordination."""

from __future__ import annotations

from wildfire_rl.routing.frontier_router import get_frontier_target
from wildfire_rl.routing.nearest_fire_router import get_nearest_fire_target
from wildfire_rl.routing.routing_utils import compute_step_action, get_sector_bounds

__all__ = [
    "get_nearest_fire_target",
    "get_frontier_target",
    "compute_step_action",
    "get_sector_bounds",
]
