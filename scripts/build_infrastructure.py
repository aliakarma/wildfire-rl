#!/usr/bin/env python
"""Build critical-infrastructure rasters for a region (Phase 15B.1).

Emits three grid-aligned ``.npy`` layers under ``data/<region>/grids/GxG/infrastructure/``:

  * ``asset_type.npy``  — int codes (0=none, 1=refinery, 2=pipeline, 3=storage, 4=industrial)
  * ``criticality.npy`` — [0, 1] value map (Gaussian falloff around assets, weighted by asset value)
  * ``blast_radius.npy`` — per-asset cascade radius in cells (0 elsewhere)

These feed ``EnvConfig.infra`` (infrastructure observation channel, catastrophe penalty, cascading
detonation). Registration assumes a north-up grid (row 0 = max latitude, col 0 = min longitude)
matching the ROI used to build the state tensors.

    python scripts/build_infrastructure.py --region saudi_eastern_province --grid 32
"""

from __future__ import annotations

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.paths import data_dir, ensure_dir

logger = get_logger("build_infrastructure")

# ROI bounds used when the state tensors were built: (lon_min, lat_min, lon_max, lat_max).
ROIS: dict[str, tuple[float, float, float, float]] = {
    "saudi_eastern_province": (45.5, 23.5, 50.5, 28.5),
    "california": (-124.5, 36.5, -119.0, 41.5),
}

# Asset code -> relative economic value (matches InfraConfig.asset_values defaults).
ASSET_VALUES: dict[int, float] = {1: 10.0, 2: 4.0, 3: 6.0, 4: 3.0}
# Asset code -> cascade blast radius in cells (refineries/storage detonate wider).
ASSET_BLAST: dict[int, int] = {1: 3, 2: 2, 3: 3, 4: 2}

# Approximate public coordinates (lon, lat, asset_type) of critical assets per region.
# Saudi = Eastern-Province petroleum facilities; California = generic critical/urban facilities
# ("forest-value assets") so the transfer study is infrastructure-aware in both directions.
DEFAULT_SITES: dict[str, list[tuple[float, float, int]]] = {
    "saudi_eastern_province": [
        (49.30, 25.40, 2),  # Ghawar / Uthmaniyah — world's largest oil field (gathering/pipeline)
        (49.67, 25.93, 1),  # Abqaiq (Buqayq) — largest crude processing facility (refinery)
        (50.10, 26.40, 4),  # Dhahran / Dammam — Saudi Aramco HQ (industrial)
        (50.16, 26.64, 1),  # Ras Tanura — refinery & export terminal (refinery)
        (48.40, 25.10, 2),  # Khurais — major field (pipeline)
        (50.00, 26.55, 3),  # Qatif — field / storage
        (48.80, 28.00, 3),  # Safaniya — near-shore field / storage
    ],
    "california": [
        (-121.49, 38.58, 4),  # Sacramento — urban/industrial
        (-122.39, 40.59, 3),  # Redding — storage/critical facility
        (-121.84, 39.73, 4),  # Chico — urban/industrial
        (-124.16, 40.80, 3),  # Eureka — coastal critical facility
        (-120.66, 40.42, 1),  # Susanville-area high-value facility
        (-122.71, 38.44, 3),  # Santa Rosa — urban/storage
    ],
}


def _rowcol(lon: float, lat: float, roi: tuple[float, float, float, float], grid: int):
    lon_min, lat_min, lon_max, lat_max = roi
    col = (lon - lon_min) / (lon_max - lon_min) * (grid - 1)
    row = (lat_max - lat) / (lat_max - lat_min) * (grid - 1)  # north up
    return row, col


def build_infrastructure(
    region: str, grid: int, sites: list[tuple[float, float, int]], sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(asset_type, criticality, blast_radius)`` rasters of shape ``(grid, grid)``."""
    if region not in ROIS:
        raise ValueError(f"Unknown region '{region}'; known: {list(ROIS)}")
    roi = ROIS[region]
    yy, xx = np.mgrid[0:grid, 0:grid].astype(float)

    asset_type = np.zeros((grid, grid), dtype=np.int32)
    blast_radius = np.zeros((grid, grid), dtype=np.int32)
    crit = np.zeros((grid, grid), dtype=np.float64)

    for lon, lat, code in sites:
        row, col = _rowcol(lon, lat, roi, grid)
        ri, ci = int(round(row)), int(round(col))
        if not (0 <= ri < grid and 0 <= ci < grid):
            continue  # site outside the ROI grid
        asset_type[ri, ci] = code
        blast_radius[ri, ci] = ASSET_BLAST.get(code, 2)
        # value-weighted Gaussian so refineries produce higher criticality than pipelines
        crit += ASSET_VALUES.get(code, 1.0) * np.exp(
            -(((yy - row) ** 2 + (xx - col) ** 2) / (2.0 * sigma**2))
        )

    peak = crit.max()
    if peak > 0:
        crit /= peak
    return asset_type, crit.astype(np.float32), blast_radius


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="saudi_eastern_province")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian falloff (grid cells)")
    args = ap.parse_args()

    sites = DEFAULT_SITES.get(args.region, [])
    if not sites:
        raise SystemExit(f"No default infrastructure sites for '{args.region}'.")

    asset_type, crit, blast = build_infrastructure(args.region, args.grid, sites, args.sigma)

    out_dir = ensure_dir(
        data_dir() / args.region / "grids" / f"{args.grid}x{args.grid}" / "infrastructure"
    )
    np.save(out_dir / "asset_type.npy", asset_type)
    np.save(out_dir / "criticality.npy", crit)
    np.save(out_dir / "blast_radius.npy", blast)
    logger.info(
        "Wrote %s  assets=%d  types=%s  crit_range=[%.3f, %.3f]",
        out_dir,
        int((asset_type > 0).sum()),
        sorted({int(c) for c in asset_type[asset_type > 0]}),
        float(crit.min()),
        float(crit.max()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
