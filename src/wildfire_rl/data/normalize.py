"""Single canonical normalization function.

The same ``normalize(x)`` was copy-pasted into the ERA5/DEM/NDVI notebooks. This is
the one implementation. NOTE (documented limitation): min-max here is *per-array*,
which means it is applied per-region independently. For cross-region transfer you
should fit a shared scaler instead — see :func:`fit_shared_minmax`.
"""

from __future__ import annotations

import numpy as np


def minmax_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale ``x`` to [0, 1] using its own min/max. NaNs are treated as the min."""
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=np.nanmin(x) if np.isfinite(x).any() else 0.0)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def fit_shared_minmax(arrays: list[np.ndarray]) -> tuple[float, float]:
    """Fit a single (min, max) across multiple arrays (e.g. both regions).

    Use this to put cross-region channels on a *shared* scale so that transfer
    evaluation is not confounded by per-region normalization.
    """
    lo = min(float(np.nanmin(a)) for a in arrays)
    hi = max(float(np.nanmax(a)) for a in arrays)
    return lo, hi


def apply_minmax(x: np.ndarray, lo: float, hi: float, eps: float = 1e-8) -> np.ndarray:
    """Apply a *given* (lo, hi) scaler (from :func:`fit_shared_minmax`) to ``x``."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=lo)
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
