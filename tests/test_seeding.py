"""Seeding utility tests (ported from V1; torch/SB3 are optional in V2, so the torch
determinism test skips cleanly when torch is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_marl.reproducibility.seeding import make_rng, set_global_seed


def test_set_global_seed_returns_seed():
    assert set_global_seed(42, deterministic_torch=False) == 42


def test_set_global_seed_makes_numpy_reproducible():
    set_global_seed(1, deterministic_torch=False)
    a = np.random.rand(5)
    set_global_seed(1, deterministic_torch=False)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_make_rng_reproducible():
    r1 = make_rng(7).random(4)
    r2 = make_rng(7).random(4)
    np.testing.assert_array_equal(r1, r2)
    assert not np.array_equal(make_rng(7).random(4), make_rng(8).random(4))


def test_set_global_seed_sets_determinism_env():
    """The hardened seeder must export the deterministic-CUDA + hash env vars."""
    import os

    set_global_seed(0)
    assert os.environ.get("PYTHONHASHSEED") == "0"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_set_global_seed_makes_torch_reproducible():
    """Same seed => identical torch draws (the RL training RNG)."""
    torch = pytest.importorskip("torch")
    set_global_seed(3)
    a = torch.rand(5)
    set_global_seed(3)
    b = torch.rand(5)
    assert torch.equal(a, b)
