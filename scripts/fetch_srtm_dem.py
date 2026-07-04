#!/usr/bin/env python
"""Fetch SRTM elevation for a region ROI and build a 3-arcsec DEM mosaic (closes the
Phase-2 flat-topography data debt — V1 archived slope *magnitude* only, no DEM/aspect).

Source: the AWS Terrain Tiles open dataset (https://registry.opendata.aws/terrain-tiles/),
``skadi`` layer — 1°x1°, 1-arcsec SRTM-format HGT tiles (void-filled composite; SRTM v3 is
the land source at both ROIs' latitudes). Each tile is downloaded once into a local cache,
clamped to sea level (the composite carries ETOPO1 bathymetry offshore; neither ROI has
meaningful below-sea-level land), block-mean downsampled 3x to 3 arcsec (~90 m), mosaicked
north-up, and cropped edge-exact to the ROI. Outputs per region (untracked — ``data/**``
is gitignored — but the mosaic's SHA-256 is recorded by ``to_cell2fire.py`` in the tracked
``conversion_report.json``):

    data/<region>/raw/dem/srtm/tiles/<TILE>.hgt.gz     raw tile cache
    data/<region>/raw/dem/srtm/<name>_dem_3arcsec.npy  int16 mosaic, rows N->S, cols W->E
    data/<region>/raw/dem/srtm/<name>_dem_manifest.json  tile URLs/hashes + processing record

    python scripts/fetch_srtm_dem.py --region saudi
    python scripts/fetch_srtm_dem.py --region california
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wildfire_marl.data.to_cell2fire import REGIONS
from wildfire_marl.paths import data_dir
from wildfire_marl.reproducibility.logging_utils import file_sha256

_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{latdir}/{tile}.hgt.gz"
_HGT_N = 3601  # 1-arcsec samples per tile side (edges shared with neighbours)
_DOWNSAMPLE = 3  # 1 arcsec -> 3 arcsec (~90 m), the classic SRTM3 resolution
_PX_PER_DEG = 3600 // _DOWNSAMPLE  # 1200


def _tile_name(lat: int, lon: int) -> tuple[str, str]:
    latdir = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
    return latdir, f"{latdir}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def _fetch(url: str, dest: Path, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                dest.write_bytes(r.read())
            return
        except Exception as e:  # noqa: BLE001 — retry any transport error, then re-raise
            if attempt == retries:
                raise
            print(f"  retry {attempt}/{retries} after error: {e}")
            time.sleep(2.0 * attempt)


def _load_tile_3arcsec(gz_path: Path) -> np.ndarray:
    """Decode one skadi HGT tile -> (1200, 1200) int16 at 3 arcsec, sea-level clamped.

    HGT is big-endian int16, row 0 = north edge; the last row/col duplicate the
    neighbouring tiles' edges and are dropped before the 3x3 block mean. Clamping
    (bathymetry/voids -> 0 m) happens *before* averaging so coastal pixels are not
    dragged down by offshore depths.
    """
    raw = np.frombuffer(gzip.decompress(gz_path.read_bytes()), dtype=">i2")
    z = raw.reshape(_HGT_N, _HGT_N)[:-1, :-1].astype(np.float64)
    z = np.maximum(z, 0.0)  # SRTM void (-32768) + ocean bathymetry -> sea level
    d = _DOWNSAMPLE
    n = z.shape[0] // d
    z = z.reshape(n, d, n, d).mean(axis=(1, 3))
    return np.round(z).astype(np.int16)


def fetch_region(region: str) -> Path:
    spec = REGIONS[region]
    lon_min, lat_min, lon_max, lat_max = spec.roi
    out_dir = data_dir() / spec.dir / "raw" / "dem" / "srtm"
    tile_dir = out_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    lats = range(math.floor(lat_min), math.ceil(lat_max))  # 1° tile SW corners, S->N
    lons = range(math.floor(lon_min), math.ceil(lon_max))
    tiles: list[dict] = []
    rows = []
    for lat in sorted(lats, reverse=True):  # mosaic is built north-up
        row = []
        for lon in lons:
            latdir, name = _tile_name(lat, lon)
            url = _URL.format(latdir=latdir, tile=name)
            dest = tile_dir / f"{name}.hgt.gz"
            print(f"tile {name} ...", flush=True)
            _fetch(url, dest)
            row.append(_load_tile_3arcsec(dest))
            tiles.append({"tile": name, "url": url, "sha256_gz": file_sha256(dest)})
        rows.append(np.hstack(row))
    mosaic = np.vstack(rows)

    # Edge-exact ROI crop (tile/pixel edges are aligned to integer degrees by construction).
    top = math.ceil(lat_max)
    left = math.floor(lon_min)
    r0 = round((top - lat_max) * _PX_PER_DEG)
    r1 = round((top - lat_min) * _PX_PER_DEG)
    c0 = round((lon_min - left) * _PX_PER_DEG)
    c1 = round((lon_max - left) * _PX_PER_DEG)
    roi = mosaic[r0:r1, c0:c1]

    npy_path = out_dir / f"{region}_dem_3arcsec.npy"
    np.save(npy_path, roi)
    manifest = {
        "region": region,
        "roi_lonlat": list(spec.roi),
        "source": "AWS Terrain Tiles (skadi/SRTM HGT), s3://elevation-tiles-prod",
        "resolution_arcsec": 3,
        "mosaic_shape": list(roi.shape),
        "mosaic_dtype": "int16 (metres, rounded after 3x3 block mean)",
        "orientation": "north-up (row 0 = lat_max edge, col 0 = lon_min edge)",
        "processing": "last HGT row/col dropped (shared edges); clamp <0 m to 0 "
        "(ETOPO1 bathymetry + SRTM voids) BEFORE 3x3 block mean; edge-exact ROI crop",
        "mosaic_sha256": file_sha256(npy_path),
        "elevation_m": {
            "min": int(roi.min()),
            "max": int(roi.max()),
            "mean": round(float(roi.mean()), 1),
        },
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tiles": tiles,
    }
    (out_dir / f"{region}_dem_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"{region}: mosaic {roi.shape} -> {npy_path}\n"
        f"  elevation m: min {roi.min()} max {roi.max()} mean {roi.mean():.1f}"
    )
    return npy_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True, choices=sorted(REGIONS))
    args = ap.parse_args()
    fetch_region(args.region)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
