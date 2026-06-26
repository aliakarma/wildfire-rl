#!/usr/bin/env python
"""Train PPO (multi-seed) for a region. Thin wrapper around the library CLI.

    python scripts/train.py --config configs/experiment/multiseed.yaml
    python scripts/train.py --set region.dir=california region.name=california ppo.total_timesteps=1000
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from wildfire_rl.cli import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(["train", *sys.argv[1:]]))
