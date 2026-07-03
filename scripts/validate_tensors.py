#!/usr/bin/env python
"""Tensor invariants guard: catch silently corrupt or mis-scaled data channels.

Validates every ``state_tensor.npy`` under ``data/`` against the frozen channel contract
(``wildfire_rl.data.CHANNEL_ORDER``) and the data card's normalization guarantee (per-channel
min-max to [0, 1], see ``docs/data_card.md``):

  * shape is ``(7, H, W)`` with H == W,
  * all values finite (no NaN/Inf),
  * every channel within [0, 1] (small tolerance) — not just ``fire``.

Any Saudi ``criticality.npy`` asset raster (Phase 15) is checked for the same [0, 1] range.

Exit 0 = all OK, 1 = at least one invariant violated. CI-callable (Phase 13) and part of the
provenance surface (Phase 11).

    python scripts/validate_tensors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (adds src/ to path when not installed)
from wildfire_rl.data import CHANNEL_ORDER

TOL = 1e-6
N_CH = len(CHANNEL_ORDER)


def _check_state_tensor(path: Path) -> list[str]:
    t = np.load(path)
    errs: list[str] = []
    if t.ndim != 3 or t.shape[0] != N_CH:
        errs.append(f"{path}: expected ({N_CH}, H, W), got {t.shape}")
        return errs  # shape wrong -> per-channel checks are meaningless
    if t.shape[1] != t.shape[2]:
        errs.append(f"{path}: non-square grid {t.shape[1]}x{t.shape[2]}")
    if not np.isfinite(t).all():
        errs.append(f"{path}: contains non-finite values (NaN/Inf)")
    for i, name in enumerate(CHANNEL_ORDER):
        lo, hi = float(t[i].min()), float(t[i].max())
        if lo < -TOL or hi > 1.0 + TOL:
            errs.append(f"{path}: channel {i} '{name}' out of [0,1] (min={lo:.4f}, max={hi:.4f})")
    if not errs:
        print(f"OK  {path}  shape={t.shape}")
    return errs


def _check_range_raster(path: Path, label: str) -> list[str]:
    t = np.load(path)
    errs: list[str] = []
    if not np.isfinite(t).all():
        errs.append(f"{path}: contains non-finite values (NaN/Inf)")
    lo, hi = float(t.min()), float(t.max())
    if lo < -TOL or hi > 1.0 + TOL:
        errs.append(f"{path}: {label} out of [0,1] (min={lo:.4f}, max={hi:.4f})")
    if not errs:
        print(f"OK  {path}  ({label}, shape={t.shape})")
    return errs


def main() -> int:
    data = Path("data")
    if not data.exists():
        print("no data/ directory (skip)")
        return 0

    problems: list[str] = []
    state_tensors = sorted(data.rglob("state_tensor.npy"))
    if not state_tensors:
        print("no state_tensor.npy found under data/ (skip)")
    for p in state_tensors:
        problems += _check_state_tensor(p)

    for p in sorted(data.rglob("criticality.npy")):
        problems += _check_range_raster(p, "criticality")

    if problems:
        print("\nTENSOR VALIDATION: FAIL")
        for pr in problems:
            print("  -", pr)
        return 1
    print("\nTENSOR VALIDATION: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
