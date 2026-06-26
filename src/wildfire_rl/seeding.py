"""Reproducibility / deterministic RNG utilities.

The original notebooks seeded ``random``/``numpy``/``torch`` inconsistently (only one
notebook defined a ``set_seed``), and — critically — the environment dynamics used the
*global* ``np.random`` instead of a per-env generator, so per-episode seeds had no effect.
This module centralizes global seeding; the environment uses its own ``self.np_random``
(see :mod:`wildfire_rl.envs.base`) so dynamics are actually reproducible.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> int:
    """Seed all global RNGs (Python, NumPy, PyTorch, SB3) and return the seed.

    ``deterministic_torch`` additionally requests deterministic cuDNN kernels. This
    can slow training slightly but removes a major source of run-to-run variation.
    Torch / SB3 are imported lazily so this works in lightweight (CPU/test)
    environments without those packages installed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(seed)
    except ImportError:
        pass

    return seed


def make_rng(seed: int | None) -> np.random.Generator:
    """Return an isolated NumPy ``Generator`` (preferred over global ``np.random``)."""
    return np.random.default_rng(seed)
