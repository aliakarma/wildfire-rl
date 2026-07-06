"""Statistical significance testing for AAAI-grade results.

Provides paired t-tests, Welch's t-tests, 95% confidence intervals,
Cohen's d effect sizes, and Wilcoxon signed-rank tests. All functions
accept plain arrays so they compose with the canonical evaluation loop
(ported into ``wildfire_marl.eval.evaluate`` in Phase 4).

Ported unchanged from V1 (``legacy_v1/src/wildfire_rl/eval/significance.py``).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def confidence_interval_95(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Return 95% CI ``(lower, upper)`` using the *t*-distribution.

    For *n* < 2 the mean is returned as both bounds (degenerate interval).
    """
    a = np.asarray(values, dtype=float)
    n = len(a)
    if n < 2:
        m = float(a.mean())
        return (m, m)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + 0.95) / 2, n - 1)
    m = float(a.mean())
    return (m - h, m + h)


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Distribution-free bootstrap CI of the mean (percentile method).

    Preferred over the *t*-based CI for small *n* (e.g. 5 seeds). For *n* < 2 the mean is
    returned as both bounds. Seeded for reproducibility.
    """
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        m = float(a.mean())
        return (m, m)
    rng = np.random.default_rng(seed)
    boot = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    return (float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)))


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's *d* effect size with pooled standard deviation.

    |d| < 0.2: negligible, 0.2–0.5: small, 0.5–0.8: medium, > 0.8: large.
    """
    g1, g2 = np.asarray(group1, dtype=float), np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        # Degenerate (zero within-group variance): the effect size is undefined when the
        # means differ — NOT zero. Returning 0.0 previously mislabeled huge deterministic
        # differences as "no effect" (audit finding on the ablation table).
        return float("nan") if abs(g1.mean() - g2.mean()) > 1e-12 else 0.0
    return float((g1.mean() - g2.mean()) / pooled_std)


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------


def welch_ttest(group1: np.ndarray, group2: np.ndarray) -> dict[str, float]:
    """Welch's *t*-test (unequal variance, independent samples).

    Returns a dict with *t*-statistic, *p*-value, Cohen's *d*, per-group
    means, and 95 % confidence intervals.
    """
    g1, g2 = np.asarray(group1, dtype=float), np.asarray(group2, dtype=float)
    t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)
    ci1 = confidence_interval_95(g1)
    ci2 = confidence_interval_95(g2)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d(g1, g2),
        "group1_mean": float(g1.mean()),
        "group2_mean": float(g2.mean()),
        "group1_ci_lo": ci1[0],
        "group1_ci_hi": ci1[1],
        "group2_ci_lo": ci2[0],
        "group2_ci_hi": ci2[1],
    }


def paired_ttest(values1: np.ndarray, values2: np.ndarray) -> dict[str, float]:
    """Paired *t*-test for same-episode comparisons (e.g. native vs transfer).

    Both arrays must have the same length.
    """
    v1, v2 = np.asarray(values1, dtype=float), np.asarray(values2, dtype=float)
    t_stat, p_value = stats.ttest_rel(v1, v2)
    diff = v1 - v2
    ci = confidence_interval_95(diff)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(diff.mean()),
        "diff_ci_lo": ci[0],
        "diff_ci_hi": ci[1],
        "cohens_d": cohens_d(v1, v2),
    }


def wilcoxon_test(values1: np.ndarray, values2: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank test (non-parametric paired alternative)."""
    v1, v2 = np.asarray(values1, dtype=float), np.asarray(values2, dtype=float)
    if np.all(v1 == v2):
        # All differences are zero -- the test is undefined. Older SciPy raised
        # ValueError here (handled below); SciPy >= 1.14 instead returns (0.0, 1.0)
        # with a warning, so the guard is now explicit to preserve the V1 contract.
        return {"statistic": float("nan"), "p_value": float("nan")}
    try:
        stat, p_value = stats.wilcoxon(v1, v2)
        return {"statistic": float(stat), "p_value": float(p_value)}
    except ValueError:
        # All differences are zero — cannot compute.
        return {"statistic": float("nan"), "p_value": float("nan")}


# ---------------------------------------------------------------------------
# Formatting helpers (for paper tables)
# ---------------------------------------------------------------------------


def format_ci(values: list[float] | np.ndarray, fmt: str = ".2f") -> str:
    """Format as ``'mean [CI_lo, CI_hi]'``."""
    a = np.asarray(values, dtype=float)
    lo, hi = confidence_interval_95(a)
    return f"{a.mean():{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def format_mean_std(values: list[float] | np.ndarray, fmt: str = ".2f") -> str:
    """Format as ``'mean ± std'``."""
    a = np.asarray(values, dtype=float)
    return f"{a.mean():{fmt}} ± {a.std():{fmt}}"


def format_significance(p_value: float) -> str:
    """Return significance stars: ``***`` (*p* < 0.001), ``**`` (*p* < 0.01),
    ``*`` (*p* < 0.05), or ``n.s.`` (not significant)."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return "n.s."
