#!/usr/bin/env python
"""Stack per-channel ``.npy`` layers into a ``state_tensor.npy`` for a region/grid.

    python scripts/build_tensors.py --region saudi_eastern_province --grid 32
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from wildfire_rl.cli import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(["build-tensors", *sys.argv[1:]]))
