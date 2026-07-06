"""AAAI-grade statistics compiler for wildfire MARL evaluation.
Provides functions to compute bootstrap confidence intervals, Cohen's d,
Welch's t-tests, and mean/std for key metrics, comparing Hierarchical against baselines.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from wildfire_marl.eval.significance import bootstrap_ci, cohens_d


def compute_summary_stats(values: list[float] | np.ndarray, seed: int = 42) -> dict[str, float]:
    """Computes mean, std, and bootstrap 95% confidence intervals for a list of values."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci_lo, ci_hi = bootstrap_ci(arr, n_boot=1000, alpha=0.05, seed=seed)
    return {"mean": mean, "std": std, "ci_lo": ci_lo, "ci_hi": ci_hi}


def compare_policies(
    hier_vals: list[float] | np.ndarray, baseline_vals: list[float] | np.ndarray
) -> dict[str, float]:
    """Runs Welch's t-test and Cohen's d comparison between hierarchical and a baseline."""
    h = np.asarray(hier_vals, dtype=float)
    b = np.asarray(baseline_vals, dtype=float)

    if len(h) < 2 or len(b) < 2:
        return {"t_stat": 0.0, "p_val": 1.0, "cohens_d": 0.0}

    t_stat, p_val = stats.ttest_ind(h, b, equal_var=False)
    d = cohens_d(h, b)

    return {
        "t_stat": float(t_stat) if not np.isnan(t_stat) else 0.0,
        "p_val": float(p_val) if not np.isnan(p_val) else 1.0,
        "cohens_d": float(d) if not np.isnan(d) else 0.0,
    }
