"""Make ``wildfire_rl`` importable when running scripts from a source checkout
without ``pip install -e .``. Import this first in every script."""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
