"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_rl.config import EnvConfig
from wildfire_rl.data.tensor_stack import CHANNEL_ORDER


@pytest.fixture
def small_tensor() -> np.ndarray:
    """A tiny, valid 7-channel state tensor with a couple of ignition points."""
    rng = np.random.default_rng(0)
    g = 8
    t = rng.random((len(CHANNEL_ORDER), g, g)).astype(np.float32)
    t[0] = 0.0  # fire channel: start clean...
    t[0, g // 2, g // 2] = 1.0  # ...with one central ignition
    t[0, 1, 1] = 1.0
    return t


@pytest.fixture
def env_cfg() -> EnvConfig:
    return EnvConfig(grid_size=8, max_steps=10)
