"""Geospatial data ingestion (Phase 2): 7-channel region tensors -> Cell2Fire
landscape inputs (fuel grid, elevation, weather, ignitions) with one shared fuel
model and encoding across regions.

Only the frozen channel contract lives here in Phase 0; the converters arrive in
Phase 2. ``CHANNEL_ORDER`` is carried over verbatim from V1 so the preserved
``data/<region>/grids/*/state_tensor.npy`` tensors keep a single authoritative
interpretation.
"""

from __future__ import annotations

# Frozen 7-channel ordering (channels 0..6) — preserved from the V1 tensors.
CHANNEL_ORDER: list[str] = [
    "fire",
    "fuel",
    "wind_x",
    "wind_y",
    "terrain",
    "temperature",
    "humidity",
]

__all__ = ["CHANNEL_ORDER"]
