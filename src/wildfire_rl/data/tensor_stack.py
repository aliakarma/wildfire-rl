"""Canonical state-tensor assembly.

Consolidates ``05_california_tensor_stacking`` (and the missing Saudi equivalent) into
one region-agnostic implementation. Produces the frozen 7-channel ``(C, H, W)`` tensor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Frozen channel order (must match wildfire_rl.config.DEFAULT_CHANNELS).
CHANNEL_ORDER: list[str] = [
    "fire",
    "fuel",
    "wind_x",
    "wind_y",
    "terrain",
    "temperature",
    "humidity",
]


def load_layers(grid_dir: str | Path) -> dict[str, np.ndarray]:
    """Load each ``<channel>.npy`` layer from ``grid_dir`` (e.g. data/<region>/grids/32x32)."""
    grid_dir = Path(grid_dir)
    layers: dict[str, np.ndarray] = {}
    for name in CHANNEL_ORDER:
        f = grid_dir / f"{name}.npy"
        if not f.exists():
            raise FileNotFoundError(f"Missing channel layer: {f}")
        layers[name] = np.load(f).astype(np.float32)
    return layers


def stack_state_tensor(layers: dict[str, np.ndarray]) -> np.ndarray:
    """Stack per-channel 2D arrays into a ``(7, H, W)`` float32 tensor in CHANNEL_ORDER.

    Validates that all layers share the same 2D shape — a check the notebooks lacked.
    """
    shapes = {layers[c].shape for c in CHANNEL_ORDER}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent layer shapes: {shapes}")
    tensor = np.stack([layers[c] for c in CHANNEL_ORDER], axis=0).astype(np.float32)
    if tensor.ndim != 3 or tensor.shape[0] != len(CHANNEL_ORDER):
        raise ValueError(f"Unexpected tensor shape {tensor.shape}")
    return tensor


def build_and_save(grid_dir: str | Path, out_name: str = "state_tensor.npy") -> Path:
    """Build the state tensor from ``grid_dir`` layers, save it, and write metadata."""
    grid_dir = Path(grid_dir)
    layers = load_layers(grid_dir)
    tensor = stack_state_tensor(layers)

    out_path = grid_dir / out_name
    np.save(out_path, tensor)

    meta = {
        "channels": {str(i): name for i, name in enumerate(CHANNEL_ORDER)},
        "shape": list(tensor.shape),
        "grid_size": int(tensor.shape[-1]),
        "normalization": "0_to_1_per_channel_per_region",
        "dtype": "float32",
    }
    (grid_dir / "state_metadata.json").write_text(json.dumps(meta, indent=2))
    return out_path
