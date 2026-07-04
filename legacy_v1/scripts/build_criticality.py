#!/usr/bin/env python
"""Build a petroleum-asset *criticality* raster for a region.

Rasterizes point asset locations (e.g. Eastern-Province oil/gas facilities) onto the
model grid with a Gaussian falloff and normalizes to ``[0, 1]``. The environment uses
this map to penalize fire reaching high-value cells (``EnvConfig.criticality_weight``).

    python scripts/build_criticality.py --region saudi_eastern_province --grid 32
    python scripts/build_criticality.py --region saudi_eastern_province --grid 32 \
        --sites 49.30,25.40 49.67,25.93 --sigma 2.0

The default Saudi sites are APPROXIMATE public coordinates of major Eastern-Province
petroleum facilities; correct them for your study. Registration assumes a north-up grid
(row 0 = max latitude, col 0 = min longitude) matching the ROI used to build the tensors.
"""

from __future__ import annotations

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.paths import data_dir, ensure_dir

logger = get_logger("build_criticality")

# ROI bounds used when the state tensors were built: (lon_min, lat_min, lon_max, lat_max).
ROIS: dict[str, tuple[float, float, float, float]] = {
    "saudi_eastern_province": (45.5, 23.5, 50.5, 28.5),
    "california": (-124.5, 36.5, -119.0, 41.5),
}

# Approximate public coordinates (lon, lat) of major Eastern-Province petroleum assets.
DEFAULT_SITES: dict[str, list[tuple[float, float]]] = {
    "saudi_eastern_province": [
        (49.30, 25.40),  # Ghawar / Uthmaniyah — world's largest oil field
        (49.67, 25.93),  # Abqaiq (Buqayq) — largest crude processing facility
        (50.10, 26.40),  # Dhahran / Dammam — Saudi Aramco HQ
        (50.16, 26.64),  # Ras Tanura — refinery & export terminal
        (48.40, 25.10),  # Khurais — major field
        (50.00, 26.55),  # Qatif — field
        (48.80, 28.00),  # Safaniya — near-shore
    ],
}


def build_criticality(
    region: str, grid: int, sites: list[tuple[float, float]], sigma: float
) -> np.ndarray:
    """Return a ``(grid, grid)`` float32 criticality map in [0, 1]."""
    if region not in ROIS:
        raise ValueError(f"Unknown region '{region}'; known: {list(ROIS)}")
    lon_min, lat_min, lon_max, lat_max = ROIS[region]
    yy, xx = np.mgrid[0:grid, 0:grid].astype(float)
    crit = np.zeros((grid, grid), dtype=np.float64)
    for lon, lat in sites:
        col = (lon - lon_min) / (lon_max - lon_min) * (grid - 1)
        row = (lat_max - lat) / (lat_max - lat_min) * (grid - 1)  # north up
        crit += np.exp(-(((yy - row) ** 2 + (xx - col) ** 2) / (2.0 * sigma**2)))
    peak = crit.max()
    if peak > 0:
        crit /= peak
    return crit.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="saudi_eastern_province")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument(
        "--sites",
        nargs="*",
        default=None,
        help="lon,lat pairs; default = built-in petroleum sites for the region",
    )
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian falloff (grid cells)")
    ap.add_argument("--out", default=None, help="Output .npy path (default: grid dir)")
    args = ap.parse_args()

    if args.sites is None:
        sites = DEFAULT_SITES.get(args.region, [])
        if not sites:
            raise SystemExit(f"No default sites for '{args.region}'; pass --sites lon,lat ...")
    else:
        sites = [tuple(float(v) for v in s.split(",")) for s in args.sites]  # type: ignore[misc]

    crit = build_criticality(args.region, args.grid, sites, args.sigma)

    if args.out:
        out = args.out
    else:
        grid_dir = ensure_dir(data_dir() / args.region / "grids" / f"{args.grid}x{args.grid}")
        out = grid_dir / "criticality.npy"
    np.save(out, crit)
    logger.info(
        "Wrote %s  shape=%s  range=[%.3f, %.3f]  sites=%d",
        out,
        crit.shape,
        float(crit.min()),
        float(crit.max()),
        len(sites),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
