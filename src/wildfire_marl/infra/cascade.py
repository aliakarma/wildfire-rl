"""Post-spread cascading petroleum detonation (Phase 3).

Simulates explosion propagation entirely at the wrapper level:
if a cell containing a high-value asset burns, it can explode and ignite
surrounding burnable cells within its blast radius.
"""

from __future__ import annotations

import numpy as np

# Fire intensity threshold above which an asset counts as burning
ASSET_BURNING_THRESHOLD = 0.1


def cascade_step(
    fire_state: np.ndarray,
    asset_type: np.ndarray | None,
    blast_radius: np.ndarray | None,
    fuel_mask: np.ndarray,
    previously_detonated: set[int],
    cascade_prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, set[int], int]:
    """Apply wrapper-level cascading detonation for this step.

    For any asset cell that is burning (> 0.1) and has not detonated yet:
      * Trigger detonation and add to previously_detonated.
      * Ignite cells within its circular blast radius with probability cascade_prob
        (if the cell is fuel-available and not already burning/harvested).

    Args:
        fire_state: 2D array of shape (H, W), values: >0 fire, <0 harvested, 0 untouched.
        asset_type: 2D array of asset codes, or None.
        blast_radius: 2D array of blast radius in cells, or None.
        fuel_mask: 2D array where >0 indicates fuel.
        previously_detonated: Set of 0-indexed cell coordinates that have already detonated.
        cascade_prob: Probability [0, 1] of neighboring ignitions.
        rng: NumPy random generator.

    Returns:
        updated_fire_state: Updated fire_state array.
        newly_detonated: Set of cell indices that detonated *this step*.
        newly_ignited_count: Count of cells newly ignited by the cascade this step.
    """
    if asset_type is None or cascade_prob <= 0.0:
        return fire_state, set(), 0

    h, w = fire_state.shape
    new_fire_state = fire_state.copy()
    newly_detonated = set()
    newly_ignited_count = 0

    # Find currently burning assets
    burning_assets = np.argwhere((asset_type > 0) & (new_fire_state > ASSET_BURNING_THRESHOLD))
    
    for ax, ay in burning_assets:
        idx = int(ax * w + ay)
        if idx in previously_detonated:
            continue
        
        # Trigger detonation
        newly_detonated.add(idx)
        
        # Get blast radius for this asset
        r = int(blast_radius[ax, ay]) if blast_radius is not None else 2
        
        # Scan circular radius in grid cells
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if (dx == 0 and dy == 0) or (dx * dx + dy * dy > r * r):
                    continue
                nx, ny = int(ax) + dx, int(ay) + dy
                if 0 <= nx < h and 0 <= ny < w:
                    # Ignite only if it's a fuel cell and not already burning
                    if (
                        fuel_mask[nx, ny] > 0
                        and new_fire_state[nx, ny] <= ASSET_BURNING_THRESHOLD
                    ):
                        if rng.random() < cascade_prob:
                            new_fire_state[nx, ny] = 1.0  # Ignite at full intensity
                            newly_ignited_count += 1

    return new_fire_state, newly_detonated, newly_ignited_count
