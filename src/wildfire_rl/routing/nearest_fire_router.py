"""Nearest fire router implementation."""

from __future__ import annotations

import numpy as np

from wildfire_rl.routing.routing_utils import get_sector_bounds


def get_nearest_fire_target(
    fire_channel: np.ndarray,
    ax: int,
    ay: int,
    sector_idx: int,
    spread_threshold: float = 0.2,
) -> tuple[int, int]:
    """Find the coordinates of the nearest active fire cell within the target sector.

    If no active fire exists in the selected sector, falls back to the global nearest fire.
    If the global grid contains no fire, returns (ax, ay) to stay in place.
    """
    grid_size = fire_channel.shape[0]

    # Find burning cells
    active_mask = fire_channel > spread_threshold
    burning_indices = np.argwhere(active_mask)

    if len(burning_indices) == 0:
        return ax, ay  # Stay in place

    # Try target sector first
    row_slice, col_slice = get_sector_bounds(sector_idx, grid_size)
    sector_mask = np.zeros_like(active_mask, dtype=bool)
    sector_mask[row_slice, col_slice] = True
    
    sector_burning = np.argwhere(active_mask & sector_mask)

    # Fall back to global search if target sector has no fire
    targets = sector_burning if len(sector_burning) > 0 else burning_indices

    # Find closest target cell using Manhattan distance
    dists = np.abs(targets[:, 0] - ax) + np.abs(targets[:, 1] - ay)
    closest_idx = np.argmin(dists)
    
    tx, ty = targets[closest_idx]
    return int(tx), int(ty)
