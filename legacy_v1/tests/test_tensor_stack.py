"""Tensor stacking + normalization tests."""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_rl.data.normalize import apply_minmax, fit_shared_minmax, minmax_normalize
from wildfire_rl.data.tensor_stack import CHANNEL_ORDER, stack_state_tensor


def _layers(g=6):
    rng = np.random.default_rng(0)
    return {c: rng.random((g, g)).astype(np.float32) for c in CHANNEL_ORDER}


def test_stack_shape_and_order():
    layers = _layers()
    t = stack_state_tensor(layers)
    assert t.shape == (7, 6, 6)
    assert t.dtype == np.float32
    # channel 0 must equal the 'fire' layer
    np.testing.assert_array_equal(t[0], layers["fire"])


def test_stack_rejects_mismatched_shapes():
    layers = _layers()
    layers["fuel"] = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        stack_state_tensor(layers)


def test_minmax_normalize_range():
    x = np.array([[0.0, 5.0], [10.0, 2.0]], dtype=np.float32)
    out = minmax_normalize(x)
    assert out.min() == 0.0 and out.max() == 1.0


def test_minmax_constant_array():
    x = np.full((4, 4), 7.0, dtype=np.float32)
    out = minmax_normalize(x)
    assert np.all(out == 0.0)


def test_shared_minmax_cross_region():
    a = np.array([0.0, 10.0])
    b = np.array([5.0, 20.0])
    lo, hi = fit_shared_minmax([a, b])
    assert lo == 0.0 and hi == 20.0
    assert apply_minmax(b, lo, hi).max() == 1.0
