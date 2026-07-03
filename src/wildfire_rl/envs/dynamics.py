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
            prob = cfg.spread_scale * (
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
        spread_prob = cfg.spread_scale * (
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
    for x, y in positions:
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


def maybe_reignite(state: np.ndarray, cfg: EnvConfig, rng: np.random.Generator) -> None:
    """With probability ``cfg.ignition_rate``, ignite one random cell (in place).

    Models the higher, more-random ignition frequency of the Saudi desert regime. A no-op
    when ``ignition_rate <= 0`` (default), so other regions are unaffected.
    """
    if cfg.ignition_rate <= 0.0:
        return
    if rng.random() < cfg.ignition_rate:
        h, w = state.shape[1], state.shape[2]
        x = int(rng.integers(0, h))
        y = int(rng.integers(0, w))
        state[0, x, y] = max(float(state[0, x, y]), cfg.ignition_intensity)


def asset_penalty(criticality: np.ndarray | None, fire_channel: np.ndarray, weight: float) -> float:
    """Cost of fire overlapping high-value (e.g. petroleum) cells: ``weight * Σ(crit·fire)``.

    Returns 0.0 when there is no criticality map or the weight is non-positive.
    """
    if criticality is None or weight <= 0.0:
        return 0.0
    return float(weight * (criticality * fire_channel).sum())


def load_criticality(cfg: EnvConfig, grid_size: int) -> np.ndarray | None:
    """Load a grid-aligned criticality raster in [0, 1], or ``None`` if unset.

    Raises if the raster shape does not match the environment grid.
    """
    if not cfg.criticality_path:
        return None
    crit = np.load(cfg.criticality_path).astype(np.float32)
    if crit.shape != (grid_size, grid_size):
        raise ValueError(f"criticality shape {crit.shape} != grid ({grid_size}, {grid_size})")
    return crit


# --------------------------------------------------------------------------------------------------
# Critical / petroleum infrastructure (Phase 15B.1)
# --------------------------------------------------------------------------------------------------

# Fire above this level counts an asset cell as "burning" (matches the heuristic policies' threshold).
ASSET_BURNING_THRESHOLD = 0.1


def load_infrastructure(
    cfg: EnvConfig, grid_size: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load ``(asset_type, criticality)`` rasters from ``cfg.infra.infra_dir``.

    ``asset_type`` is an int grid of codes (0=none, 1=refinery, 2=pipeline, 3=storage, 4=industrial);
    ``criticality`` is a [0, 1] value map (may be ``None`` if not present). Returns ``(None, None)``
    when no infra dir is configured. Raises on shape mismatch.
    """
    infra = cfg.infra
    if not infra.infra_dir:
        return None, None
    from pathlib import Path

    d = Path(infra.infra_dir)
    asset_type = np.load(d / "asset_type.npy").astype(np.int32)
    if asset_type.shape != (grid_size, grid_size):
        raise ValueError(f"asset_type shape {asset_type.shape} != grid ({grid_size}, {grid_size})")
    crit_path = d / "criticality.npy"
    criticality = None
    if crit_path.exists():
        criticality = np.load(crit_path).astype(np.float32)
        if criticality.shape != (grid_size, grid_size):
            raise ValueError(
                f"infra criticality shape {criticality.shape} != grid ({grid_size}, {grid_size})"
            )
    return asset_type, criticality


def cascade_explosion(
    state: np.ndarray,
    asset_type: np.ndarray | None,
    infra,  # InfraConfig (avoid a circular import annotation)
    rng: np.random.Generator,
) -> int:
    """Cascading petroleum detonation: a burning asset cell ignites cells within ``blast_radius``.

    Each cell inside the (circular) blast radius of a burning asset is ignited with probability
    ``cascade_prob``. Applied in place after fire spread; returns the number of cells newly ignited.
    No-op when ``cascade_prob <= 0`` or there are no assets, so other regions are unaffected. Uses
    the passed-in ``rng`` for reproducibility.
    """
    if asset_type is None or infra.cascade_prob <= 0.0:
        return 0
    fire = state[0]
    h, w = fire.shape
    r = int(infra.blast_radius)
    burning_assets = np.argwhere((asset_type > 0) & (fire > ASSET_BURNING_THRESHOLD))
    n_new = 0
    for ax, ay in burning_assets:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if (dx == 0 and dy == 0) or (dx * dx + dy * dy > r * r):
                    continue
                nx, ny = int(ax) + dx, int(ay) + dy
                if (
                    0 <= nx < h
                    and 0 <= ny < w
                    and fire[nx, ny] <= ASSET_BURNING_THRESHOLD
                    and rng.random() < infra.cascade_prob
                ):
                    fire[nx, ny] = 1.0  # detonation ignites at full intensity
                    n_new += 1
    return n_new


def catastrophe_penalty(
    asset_type: np.ndarray | None,
    fire_channel: np.ndarray,
    asset_values: dict[int, float],
    weight: float,
) -> float:
    """Risk-weighted cost of fire reaching assets: ``weight * Σ_cells value[type] · fire``.

    Higher-value assets (refineries) dominate, so the agent is driven to defend them preferentially
    rather than merely minimize total burned cells. Returns 0.0 when disabled or asset-free.
    """
    if asset_type is None or weight <= 0.0:
        return 0.0
    total = 0.0
    for code, val in asset_values.items():
        mask = asset_type == int(code)
        if mask.any():
            total += float(val) * float(fire_channel[mask].sum())
    return float(weight * total)
