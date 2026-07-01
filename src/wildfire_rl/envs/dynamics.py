"""Pure wildfire dynamics functions, shared by the single- and multi-agent envs.

Extracted so there is exactly ONE implementation of spread / suppression / decay,
rather than six near-identical copies across notebooks. All functions are pure
(operate on arrays + an explicit RNG) and use the passed-in ``numpy.random.Generator``
so behavior is reproducible (the original code used the unseeded global ``np.random``).
"""

from __future__ import annotations

import numpy as np

from wildfire_rl.config import EnvConfig

# Orthogonal neighborhood (matches the original 4-neighbor spread).
_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _shift(arr: np.ndarray, di: int, dj: int) -> np.ndarray:
    """Shift a 2D array by (di, dj), zero-filling the exposed border."""
    out = np.zeros_like(arr)
    h, w = arr.shape
    src_i = slice(max(0, -di), h - max(0, di))
    src_j = slice(max(0, -dj), w - max(0, dj))
    dst_i = slice(max(0, di), h - max(0, -di))
    dst_j = slice(max(0, dj), w - max(0, -dj))
    out[dst_i, dst_j] = arr[src_i, src_j]
    return out


def spread_fire(state: np.ndarray, cfg: EnvConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply one step of stochastic fire spread; returns the new fire channel.

    Probability that a cell ignites (when it has >=1 burning orthogonal neighbor):
        p = base + fuel_coeff*fuel + wind_coeff*wind_factor + terrain_coeff*terrain
    where ``wind_factor`` is the isotropic ``(wind_x + wind_y)/2`` (original) or the
    magnitude ``hypot(wind_x, wind_y)`` when ``cfg.directional_wind`` is set.

    This is a vectorized, seeded reimplementation: one Bernoulli draw per target cell
    (the original drew once per (source, neighbor) pair via the unseeded global RNG).
    """
    fire = state[0]
    fuel = state[1]
    wind_x, wind_y = state[2], state[3]
    terrain = state[4]

    active = (fire > cfg.spread_threshold).astype(np.float32)

    if cfg.directional_wind:
        ignite = np.zeros(fire.shape, dtype=bool)
        for di, dj in _OFFSETS:
            shifted = _shift(active, di, dj)
            # The fire spreads from neighbor (x-di, y-dj) to target (x, y).
            # The direction vector is (di, dj). Dot product with wind vector:
            proj = wind_x * di + wind_y * dj
            prob = (
                cfg.base_spread
                + cfg.fuel_coeff * fuel
                + cfg.wind_coeff * proj
                + cfg.terrain_coeff * terrain
            )
            prob = np.clip(prob, 0.0, 1.0)
            draws = rng.random(fire.shape)
            ignite |= (shifted > 0) & (draws < prob)
    else:
        burning_neighbors = sum(_shift(active, di, dj) for di, dj in _OFFSETS)
        can_ignite = burning_neighbors > 0
        wind_factor = (wind_x + wind_y) / 2.0
        spread_prob = (
            cfg.base_spread
            + cfg.fuel_coeff * fuel
            + cfg.wind_coeff * wind_factor
            + cfg.terrain_coeff * terrain
        )
        draws = rng.random(fire.shape)
        ignite = can_ignite & (draws < spread_prob)

    new_fire = fire.copy()
    new_fire[ignite] = np.minimum(1.0, new_fire[ignite] + cfg.fire_increment)
    return new_fire


def apply_suppression(state: np.ndarray, positions: list[tuple[int, int]], cfg: EnvConfig) -> None:
    """Multiply fire by ``suppression_factor`` in the patch around each agent (in place)."""
    h, w = state.shape[1], state.shape[2]
    r = cfg.suppression_radius
    for (x, y) in positions:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    state[0, nx, ny] *= cfg.suppression_factor


def decay_and_deplete(state: np.ndarray, cfg: EnvConfig) -> None:
    """Apply fire decay, extinguish tiny fires, and deplete fuel (in place)."""
    state[0] *= cfg.decay
    np.clip(state[0], 0.0, 1.0, out=state[0])
    state[0][state[0] < cfg.extinguish_threshold] = 0.0
    state[1] -= state[0] * cfg.fuel_depletion
    np.clip(state[1], 0.0, 1.0, out=state[1])


def randomize_ignition(
    base_tensor: np.ndarray, cfg: EnvConfig, rng: np.random.Generator
) -> np.ndarray:
    """Return a copy of ``base_tensor`` with a fresh, randomized fire channel.

    Enables a *distribution* of scenarios (disjoint train/eval ignition maps),
    which is the prerequisite for measuring generalization rather than memorization.
    """
    tensor = base_tensor.copy()
    tensor[0] = 0.0
    h, w = tensor.shape[1], tensor.shape[2]
    n = max(1, cfg.n_ignition_points)
    xs = rng.integers(0, h, size=n)
    ys = rng.integers(0, w, size=n)
    tensor[0, xs, ys] = cfg.ignition_intensity
    return tensor
