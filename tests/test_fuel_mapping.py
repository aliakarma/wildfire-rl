"""Shared NDVI→FBP fuel-mapping tests (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_marl.data.fuel_mapping import (
    NDVI_FBP_BINS,
    NONFUEL_CODE,
    class_distribution,
    code_to_fueltype,
    fuel_channel_to_ndvi,
    ndvi_to_fbp_codes,
)


def test_bins_are_monotone_and_cover_all_ndvi():
    uppers = [b[0] for b in NDVI_FBP_BINS]
    assert uppers == sorted(uppers), "bin edges must be increasing"
    assert uppers[-1] == np.inf, "last bin must be open-ended"
    # every representable NDVI maps to exactly one code
    ndvi = np.linspace(-1.0, 1.0, 401)
    codes = ndvi_to_fbp_codes(ndvi)
    valid = {b[1] for b in NDVI_FBP_BINS}
    assert set(np.unique(codes)) <= valid


def test_known_values_map_to_expected_classes():
    # barren -> NF, sparse -> O1a, grass -> O1b, open woodland -> C7, dense -> C3
    codes = ndvi_to_fbp_codes(np.array([-0.2, 0.0, 0.1, 0.2, 0.35, 0.5, 0.8]))
    assert list(codes) == [101, 101, 31, 32, 7, 5, 3]


def test_mapping_is_region_independent():
    """The load-bearing property: the SAME function of NDVI for both regions."""
    ndvi = np.array([[0.02, 0.12], [0.4, 0.7]])
    assert np.array_equal(ndvi_to_fbp_codes(ndvi), ndvi_to_fbp_codes(ndvi.copy()))
    # No region argument exists on the mapping — only on the NDVI *recovery* step.


def test_saudi_ndvi_recovery_is_exact_inverse():
    # V1 recorded transform: fuel = (NDVI + 1) / 2  ->  NDVI = 2*fuel - 1
    ndvi = np.array([-0.2, 0.0, 0.094, 0.5])
    fuel = (ndvi + 1.0) / 2.0
    np.testing.assert_allclose(fuel_channel_to_ndvi("saudi", fuel), ndvi, atol=1e-12)


def test_california_recovery_uses_declared_assumed_range():
    from wildfire_marl.data.fuel_mapping import ASSUMED_CA_NDVI_RANGE

    lo, hi = ASSUMED_CA_NDVI_RANGE
    fuel = np.array([0.0, 1.0])
    out = fuel_channel_to_ndvi("california", fuel)
    np.testing.assert_allclose(out, [lo, hi], atol=1e-12)


def test_unknown_region_rejected():
    with pytest.raises(ValueError):
        fuel_channel_to_ndvi("mars", np.zeros(3))


def test_code_to_fueltype_and_distribution():
    assert code_to_fueltype(NONFUEL_CODE) == "NF"
    assert code_to_fueltype(31) == "O1a"
    dist = class_distribution(np.array([[101, 31], [31, 3]]))
    assert dist == {"NF": 0.25, "O1a": 0.5, "C3": 0.25}
