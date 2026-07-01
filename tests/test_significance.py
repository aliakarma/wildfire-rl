"""Tests for the statistical significance module."""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_rl.eval.significance import (
    cohens_d,
    confidence_interval_95,
    format_ci,
    format_mean_std,
    format_significance,
    paired_ttest,
    welch_ttest,
    wilcoxon_test,
)


class TestConfidenceInterval:
    def test_ci_contains_mean(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = confidence_interval_95(vals)
        assert lo < np.mean(vals) < hi

    def test_ci_single_value(self):
        lo, hi = confidence_interval_95([42.0])
        assert lo == hi == 42.0

    def test_ci_narrow_for_tight_data(self):
        vals = [10.0, 10.01, 9.99, 10.0, 10.0]
        lo, hi = confidence_interval_95(vals)
        assert hi - lo < 0.1  # very tight

    def test_ci_wider_for_spread_data(self):
        vals = [0.0, 50.0, 100.0, 150.0, 200.0]
        lo, hi = confidence_interval_95(vals)
        assert hi - lo > 50  # wide


class TestCohensD:
    def test_identical_groups(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cohens_d(a, a) == 0.0

    def test_large_effect(self):
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        d = cohens_d(a, b)
        assert d > 0.8, f"Expected large effect, got d={d}"

    def test_sign_direction(self):
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([0.0, 1.0, 2.0])
        assert cohens_d(a, b) > 0
        assert cohens_d(b, a) < 0

    def test_degenerate_returns_nan(self):
        import math

        # Zero within-group variance but different means => effect size undefined, NOT 0.
        assert math.isnan(cohens_d(np.array([1.0, 1.0, 1.0]), np.array([9.0, 9.0, 9.0])))
        # Zero variance AND equal means => 0.0.
        assert cohens_d(np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])) == 0.0


class TestWelchTTest:
    def test_highly_significant(self):
        a = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = welch_ttest(a, b)
        assert result["p_value"] < 0.001
        assert result["cohens_d"] > 0.8
        assert "group1_ci_lo" in result
        assert result["group1_ci_lo"] < result["group1_ci_hi"]

    def test_not_significant_same_distribution(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=100)
        b = rng.normal(0, 1, size=100)
        result = welch_ttest(a, b)
        assert result["p_value"] > 0.05

    def test_all_keys_present(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = welch_ttest(a, b)
        expected_keys = {
            "t_statistic",
            "p_value",
            "cohens_d",
            "group1_mean",
            "group2_mean",
            "group1_ci_lo",
            "group1_ci_hi",
            "group2_ci_lo",
            "group2_ci_hi",
        }
        assert set(result.keys()) == expected_keys


class TestPairedTTest:
    def test_significant_difference(self):
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        result = paired_ttest(a, b)
        assert result["p_value"] < 0.001
        assert result["mean_diff"] == pytest.approx(5.0)

    def test_no_difference(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paired_ttest(a, a)
        # p-value should be NaN or 1.0 for identical arrays
        assert result["mean_diff"] == pytest.approx(0.0)


class TestWilcoxon:
    def test_significant(self):
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = wilcoxon_test(a, b)
        assert result["p_value"] < 0.05

    def test_identical_returns_nan(self):
        a = np.array([1.0, 2.0, 3.0])
        result = wilcoxon_test(a, a)
        assert np.isnan(result["p_value"])


class TestFormatting:
    def test_format_ci(self):
        s = format_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        assert "[" in s and "]" in s
        assert "3.00" in s  # mean

    def test_format_mean_std(self):
        s = format_mean_std([1.0, 2.0, 3.0])
        assert "±" in s

    def test_format_significance_stars(self):
        assert format_significance(0.0001) == "***"
        assert format_significance(0.005) == "**"
        assert format_significance(0.03) == "*"
        assert format_significance(0.1) == "n.s."
