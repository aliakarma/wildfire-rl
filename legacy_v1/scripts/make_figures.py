#!/usr/bin/env python
"""Regenerate paper figures from canonical result CSVs.

    python scripts/make_figures.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from wildfire_rl.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["make-figures"]))
