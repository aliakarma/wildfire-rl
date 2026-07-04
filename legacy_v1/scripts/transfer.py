#!/usr/bin/env python
"""Compute the full cross-region transfer matrix.

    python scripts/transfer.py --config configs/experiment/transfer.yaml
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from wildfire_rl.cli import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(["transfer", *sys.argv[1:]]))
