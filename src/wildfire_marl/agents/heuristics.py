"""Information-matched heuristic suppression policies for single-agent env (Phase 4).

All heuristics read only the observation array/dictionary (no oracle privilege)
and return a cell index in the Discrete action space.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, convolve


class BaseHeuristicPolicy:
    """ABC for heuristics, exposing SB3-like `predict` interface."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        """Predict the next action cell index to treat.

        Args:
            obs: Observation tensor of shape (3, H, W) or (4, H, W).
            deterministic: Ignored for deterministic heuristics.

        Returns:
            action: 0-indexed cell ID.
            state: Dummy state (None).
        """
        raise NotImplementedError


class NoOpPolicy(BaseHeuristicPolicy):
    """Policy that acts as a no-op (treats a non-fuel or already treated cell)."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        _, h, w = obs.shape
        fuel_mask = obs[2]

        # Treat first non-fuel cell (this has no effect on fire spread)
        non_fuel = np.flatnonzero(fuel_mask == 0)
        if non_fuel.size > 0:
            return int(non_fuel[0]), None

        # Treat already treated cell
        harvested = np.flatnonzero(obs[1] > 0)
        if harvested.size > 0:
            return int(harvested[0]), None

        return 0, None


class RandomPolicy(BaseHeuristicPolicy):
    """Policy that treats a random treatable cell."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        _, h, w = obs.shape
        fire = obs[0]
        harvested = obs[1]
        fuel_mask = obs[2]

        treatable = np.flatnonzero((fuel_mask > 0) & (fire == 0) & (harvested == 0))
        if treatable.size > 0:
            # We use standard numpy choice; evaluation seeds will keep it reproducible
            return int(np.random.choice(treatable)), None
        return 0, None


def _get_candidates(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Helper to return candidate cells, treatable mask, and fire centroid."""
    _, h, w = obs.shape
    fire = obs[0]
    harvested = obs[1]
    fuel_mask = obs[2]

    burning_mask = fire > 0
    treatable_mask = (fuel_mask > 0) & (fire == 0) & (harvested == 0)

    # Fire centroid (mean burning coordinates)
    burning_coords = np.argwhere(burning_mask)
    if len(burning_coords) > 0:
        cy, cx = burning_coords.mean(axis=0)
    else:
        cy, cx = h / 2.0, w / 2.0

    # Frontier cells: treatable cells adjacent to burning cells
    if burning_mask.any():
        frontier_mask = binary_dilation(burning_mask) & treatable_mask
        candidates = np.argwhere(frontier_mask)
    else:
        candidates = np.array([])

    # Fallback to any treatable cells if frontier is empty
    if len(candidates) == 0:
        candidates = np.argwhere(treatable_mask)

    return candidates, treatable_mask, (cy, cx)


class NearestFrontierPolicy(BaseHeuristicPolicy):
    """Treats the frontier cell closest to the fire centroid."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        _, h, w = obs.shape
        candidates, _, centroid = _get_candidates(obs)

        if len(candidates) == 0:
            return 0, None

        cy, cx = centroid
        dists = np.sum((candidates - np.array([cy, cx])) ** 2, axis=1)
        closest_idx = np.argmin(dists)
        ty, tx = candidates[closest_idx]
        return int(ty * w + tx), None


class GreatestRiskFirstPolicy(BaseHeuristicPolicy):
    """Treats the frontier cell adjacent to the most active burning neighbors."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        _, h, w = obs.shape
        candidates, _, centroid = _get_candidates(obs)

        if len(candidates) == 0:
            return 0, None

        # Count burning neighbors for each cell in 4-neighborhood
        fire = obs[0]
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        burning_neighbors = convolve((fire > 0).astype(int), kernel, mode="constant", cval=0)

        # Get neighbor counts for candidates
        counts = burning_neighbors[candidates[:, 0], candidates[:, 1]]
        max_count = counts.max()
        best_candidates = candidates[counts == max_count]

        # Tie-breaker: closest to centroid
        cy, cx = centroid
        dists = np.sum((best_candidates - np.array([cy, cx])) ** 2, axis=1)
        closest_idx = np.argmin(dists)
        ty, tx = best_candidates[closest_idx]
        return int(ty * w + tx), None


class ValueWeightedFirstPolicy(BaseHeuristicPolicy):
    """Prioritizes frontier cells with the highest infrastructure criticality values."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        _, h, w = obs.shape
        candidates, _, centroid = _get_candidates(obs)

        if len(candidates) == 0:
            return 0, None

        # If criticality channel (channel 3) is present
        if obs.shape[0] >= 4:
            crit = obs[3]
            crit_vals = crit[candidates[:, 0], candidates[:, 1]]
            max_crit = crit_vals.max()
            best_candidates = candidates[crit_vals == max_crit] if max_crit > 0.0 else candidates
        else:
            best_candidates = candidates

        # Tie-breaker: greatest risk (burning neighbors)
        fire = obs[0]
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        burning_neighbors = convolve((fire > 0).astype(int), kernel, mode="constant", cval=0)

        counts = burning_neighbors[best_candidates[:, 0], best_candidates[:, 1]]
        max_count = counts.max()
        risk_candidates = best_candidates[counts == max_count]

        # Second tie-breaker: closest to centroid
        cy, cx = centroid
        dists = np.sum((risk_candidates - np.array([cy, cx])) ** 2, axis=1)
        closest_idx = np.argmin(dists)
        ty, tx = risk_candidates[closest_idx]
        return int(ty * w + tx), None
