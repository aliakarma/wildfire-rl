"""Routing utility functions for hybrid path-planning."""

from __future__ import annotations

import numpy as np


def get_sector_bounds(sector_idx: int, grid_size: int = 32) -> tuple[slice, slice]:
    """Return the row and col slices for a given quadrant sector index.

    Sectors:
        0: North-West (NW)
        1: North-East (NE)
        2: South-West (SW)
        3: South-East (SE)
        4: Global Grid (fallback)
    """
    half = grid_size // 2
    if sector_idx == 0:
        return slice(0, half), slice(0, half)
    elif sector_idx == 1:
        return slice(0, half), slice(half, grid_size)
    elif sector_idx == 2:
        return slice(half, grid_size), slice(0, half)
    elif sector_idx == 3:
        return slice(half, grid_size), slice(half, grid_size)
    else:
        return slice(0, grid_size), slice(0, grid_size)


def compute_step_action(
    ax: int, ay: int, tx: int, ty: int, grid_size: int = 32
) -> int:
    """Return movement step action (0-4) towards the target cell (tx, ty).

    Step Actions:
        0: Move Up (row - 1)
        1: Move Down (row + 1)
        2: Move Left (col - 1)
        3: Move Right (col + 1)
        4: Stay (suppress in place)
    """
    if ax == tx and ay == ty:
        return 4  # stay

    dx = tx - ax
    dy = ty - ay

    # Move along the axis of greater distance
    if abs(dx) >= abs(dy) and dx != 0:
        return 0 if dx < 0 else 1
    elif dy != 0:
        return 2 if dy < 0 else 3
    
    return 4  # stay
