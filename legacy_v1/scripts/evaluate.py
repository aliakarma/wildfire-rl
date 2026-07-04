#!/usr/bin/env python
"""Evaluate baselines (+ optional PPO model) for a region.

    python scripts/evaluate.py --config configs/experiment/multiseed.yaml \
        --model models/ppo_saudi_32_seed_0.zip
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from wildfire_rl.cli import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
