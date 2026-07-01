#!/usr/bin/env python
"""Fix tensor dtype inconsistencies and generate missing metadata.

Phase 1 of AAAI readiness: the Saudi tensor was stored as float64 (from an
older preprocessing pipeline) while California is float32. This normalizes
both to float32 and writes state_metadata.json where missing.

    python scripts/fix_tensors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHANNEL_ORDER = ["fire", "fuel", "wind_x", "wind_y", "terrain", "temperature", "humidity"]


def fix_tensor(grid_dir: Path) -> None:
    tensor_path = grid_dir / "state_tensor.npy"
    if not tensor_path.exists():
        print(f"  SKIP: {tensor_path} not found")
        return

    t = np.load(tensor_path)
    print(f"  Before: shape={t.shape}, dtype={t.dtype}, bytes={t.nbytes}")

    if t.dtype != np.float32:
        t = t.astype(np.float32)
        np.save(tensor_path, t)
        print(f"  After:  shape={t.shape}, dtype={t.dtype}, bytes={t.nbytes}")
    else:
        print("  Already float32, no change needed.")

    # Write metadata if missing
    meta_path = grid_dir / "state_metadata.json"
    meta = {
        "channels": {str(i): name for i, name in enumerate(CHANNEL_ORDER)},
        "shape": list(t.shape),
        "grid_size": int(t.shape[-1]),
        "normalization": "0_to_1_per_channel_per_region",
        "dtype": "float32",
        "per_channel_ranges": {
            name: {"min": float(t[i].min()), "max": float(t[i].max())}
            for i, name in enumerate(CHANNEL_ORDER)
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Wrote metadata -> {meta_path}")


def main() -> int:
    data = Path("data")
    for region in ["saudi_eastern_province", "california"]:
        for grid in ["32x32", "64x64"]:
            d = data / region / "grids" / grid
            if d.exists():
                print(f"\n{d}:")
                fix_tensor(d)
            else:
                print(f"\n{d}: directory not found, skipping.")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
