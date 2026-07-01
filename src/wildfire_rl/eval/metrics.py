"""Canonical metric definitions — the single source of truth.

The notebooks counted "burned cells" with threshold ``> 0.5`` in training but ``> 0.2``
in evaluation, so numbers were not comparable across files. Here the threshold is an
explicit parameter (default 0.5, from :class:`MetricsConfig`) and every caller uses it.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BURNED_THRESHOLD = 0.5


def burned_cells(state: np.ndarray, threshold: float = DEFAULT_BURNED_THRESHOLD) -> int:
    """Number of cells whose fire intensity exceeds ``threshold``.

    Accepts a full ``(C, H, W)`` state tensor or a 2D fire channel.
    """
    fire = state[0] if state.ndim == 3 else state
    return int(np.sum(fire > threshold))


def fire_intensity(state: np.ndarray) -> float:
    """Total fire intensity (sum over the fire channel)."""
    fire = state[0] if state.ndim == 3 else state
    return float(np.sum(fire))


def containment_rate(initial_fire_total: float, final_state: np.ndarray) -> float:
    """Fraction of the initial fire mass no longer burning at episode end (higher = better).

    An agent-influenceable outcome metric, unlike raw ``fire_intensity`` (dominated by fire the
    single agent cannot reach). Clipped to ``[0, 1]``.
    """
    fire = final_state[0] if final_state.ndim == 3 else final_state
    final = float(np.sum(fire))
    denom = initial_fire_total or 1.0
    return float(min(1.0, max(0.0, 1.0 - final / denom)))


def episode_return(rewards: list[float]) -> float:
    return float(np.sum(rewards))


def summarize(episodes: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-episode metric dicts into mean/std for each key."""
    if not episodes:
        return {}
    keys = episodes[0].keys()
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([e[k] for e in episodes], dtype=float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    return out
