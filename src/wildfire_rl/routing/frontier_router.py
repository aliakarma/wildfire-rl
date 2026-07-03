"""Frontier fire router implementation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from wildfire_rl.routing.nearest_fire_router import get_nearest_fire_target
from wildfire_rl.routing.routing_utils import get_sector_bounds

_NEIGHBOR_KERNEL = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def get_frontier_target(
    fire_channel: np.ndarray,
    ax: int,
    ay: int,
    sector_idx: int,
    spread_threshold: float = 0.2,
) -> tuple[int, int]:
    """Find the coordinates of the nearest active fire frontier cell within the target sector.

    Frontier cell = active fire cell with < 4 active neighbors (marginal propagation front).
    If no frontier cell exists in the sector, falls back to the global frontier.
    If the global grid contains no frontier cells, falls back to nearest fire routing.
    """
    grid_size = fire_channel.shape[0]

    # Find burning cells
    active_mask = fire_channel > spread_threshold
    if not active_mask.any():
        return ax, ay  # Stay

    # Calculate active neighbors for each cell
    active_neighbors = convolve(
        active_mask.astype(np.int32), _NEIGHBOR_KERNEL, mode="constant", cval=0
    )

    # Frontier cells are active and have < 4 active neighbors
    frontier_mask = active_mask & (active_neighbors < 4)
    frontier_indices = np.argwhere(frontier_mask)

    if len(frontier_indices) == 0:
        # Fall back to nearest active fire cells globally
        return get_nearest_fire_target(fire_channel, ax, ay, sector_idx, spread_threshold)

    # Try target sector first
    row_slice, col_slice = get_sector_bounds(sector_idx, grid_size)
    sector_mask = np.zeros_like(frontier_mask, dtype=bool)
    sector_mask[row_slice, col_slice] = True

    sector_frontier = np.argwhere(frontier_mask & sector_mask)

    # Fall back to global frontier search if target sector has no frontier cells
    targets = sector_frontier if len(sector_frontier) > 0 else frontier_indices

    # Find closest target using Manhattan distance
    dists = np.abs(targets[:, 0] - ax) + np.abs(targets[:, 1] - ay)
    closest_idx = np.argmin(dists)

    tx, ty = targets[closest_idx]
    return int(tx), int(ty)
