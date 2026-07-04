"""Build critical-infrastructure rasters for Saudi and California (Phase 3).

Rasterizes real asset coordinates into grid-aligned ASCII (.asc) and numpy (.npy) layers
under `data/cell2fire/<Region>/` (e.g. `Saudi` and `California`).

Files emitted:
  * `asset_type.asc` / `asset_type.npy` - 0=none, 1=refinery, 2=pipeline, 3=storage, 4=industrial
  * `criticality.asc` / `criticality.npy` - [0, 1] continuous value map (Gaussian falloff)
  * `blast_radius.asc` / `blast_radius.npy` - per-asset cascade blast radius in cells
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from wildfire_marl.paths import data_dir, ensure_dir

# ROIs matching to_cell2fire.py and the state tensors
ROIS: dict[str, tuple[float, float, float, float]] = {
    "saudi": (45.5, 23.5, 50.5, 28.5),
    "california": (-124.5, 36.5, -119.0, 41.5),
}

# Mapping region aliases
REGION_FOLDERS: dict[str, str] = {
    "saudi": "Saudi",
    "california": "California",
}

# Asset values matching docs/infra_card.md
ASSET_VALUES: dict[int, float] = {
    1: 10.0,  # Refinery
    2: 4.0,   # Pipeline
    3: 6.0,   # Storage
    4: 3.0    # Industrial/Urban
}

# Asset blast radius (in cells)
ASSET_BLAST: dict[int, int] = {
    1: 3,  # Refinery
    2: 2,  # Pipeline
    3: 3,  # Storage
    4: 2   # Industrial
}

# Public coordinates of assets (lon, lat, asset_type)
SITES: dict[str, list[tuple[float, float, int]]] = {
    "saudi": [
        (49.30, 25.40, 2),  # Ghawar / Uthmaniyah (pipeline)
        (49.67, 25.93, 1),  # Abqaiq (refinery)
        (50.10, 26.40, 4),  # Dhahran / Dammam (industrial)
        (50.16, 26.64, 1),  # Ras Tanura (refinery)
        (48.40, 25.10, 2),  # Khurais (pipeline)
        (50.00, 26.55, 3),  # Qatif (storage)
        (48.80, 28.00, 3)   # Safaniya (storage)
    ],
    "california": [
        (-121.49, 38.58, 4),  # Sacramento (industrial)
        (-122.39, 40.59, 3),  # Redding (storage)
        (-121.84, 39.73, 4),  # Chico (industrial)
        (-124.16, 40.80, 3),  # Eureka (storage)
        (-120.66, 40.42, 1),  # Susanville-area (refinery)
        (-122.71, 38.44, 3)   # Santa Rosa (storage)
    ]
}


def _rowcol(lon: float, lat: float, roi: tuple[float, float, float, float], grid: int) -> tuple[int, int]:
    """Determine the row/col cell indexes for a given coordinate."""
    lon_min, lat_min, lon_max, lat_max = roi
    # Standard cell binning aligned with the cell boundaries of the grid
    c = int(np.clip(np.floor((lon - lon_min) / (lon_max - lon_min) * grid), 0, grid - 1))
    r = int(np.clip(np.floor((lat_max - lat) / (lat_max - lat_min) * grid), 0, grid - 1))
    return r, c


def _write_asc(path: Path, grid_values: np.ndarray, cellsize_m: float = 100.0, fmt: str = "%d") -> None:
    """Write an ESRI ASCII grid file."""
    h, w = grid_values.shape
    lines = [
        f"ncols {w}",
        f"nrows {h}",
        "xllcorner 0",
        "yllcorner 0",
        f"cellsize {cellsize_m}",
        "NODATA_value -9999",
    ]
    for row in grid_values:
        lines.append(" ".join(fmt % v for v in row))
    path.write_text("\n".join(lines) + "\n")


def build_infrastructure(
    region: str,
    grid: int,
    sigma: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute infrastructure grids: asset_type, criticality, blast_radius."""
    if region not in ROIS:
        raise ValueError(f"Unknown region: {region}. Supported: {list(ROIS.keys())}")
    
    roi = ROIS[region]
    sites = SITES[region]
    
    asset_type = np.zeros((grid, grid), dtype=np.int32)
    blast_radius = np.zeros((grid, grid), dtype=np.int32)
    crit = np.zeros((grid, grid), dtype=np.float64)
    
    yy, xx = np.mgrid[0:grid, 0:grid].astype(float)
    
    for lon, lat, code in sites:
        r, c = _rowcol(lon, lat, roi, grid)
        asset_type[r, c] = code
        blast_radius[r, c] = ASSET_BLAST.get(code, 2)
        # Value-weighted Gaussian falloff
        weight = ASSET_VALUES.get(code, 1.0)
        crit += weight * np.exp(-(((yy - r) ** 2 + (xx - c) ** 2) / (2.0 * sigma**2)))
        
    peak = crit.max()
    if peak > 0:
        crit /= peak
        
    return asset_type, crit.astype(np.float32), blast_radius


def main() -> None:
    parser = argparse.ArgumentParser(description="Build infrastructure rasters.")
    parser.add_argument("--region", required=True, choices=["saudi", "california"], help="Region code.")
    parser.add_argument("--grid", type=int, default=32, help="Grid size.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian falloff sigma.")
    args = parser.parse_args()

    out_folder = data_dir() / "cell2fire" / REGION_FOLDERS[args.region]
    ensure_dir(out_folder)
    
    asset_type, criticality, blast_radius = build_infrastructure(args.region, args.grid, args.sigma)
    
    # Save ASCII files (.asc) for the Cell2Fire folder
    _write_asc(out_folder / "asset_type.asc", asset_type, fmt="%d")
    _write_asc(out_folder / "criticality.asc", criticality, fmt="%.4f")
    _write_asc(out_folder / "blast_radius.asc", blast_radius, fmt="%d")
    
    # Save numpy files (.npy) for fast loading
    np.save(out_folder / "asset_type.npy", asset_type)
    np.save(out_folder / "criticality.npy", criticality)
    np.save(out_folder / "blast_radius.npy", blast_radius)
    
    print(f"Successfully wrote asset_type, criticality, blast_radius (.asc & .npy) to {out_folder}")


if __name__ == "__main__":
    main()
