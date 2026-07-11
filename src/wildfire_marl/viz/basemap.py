"""OpenStreetMap basemap tiles for rollout rendering.

Fetches OSM tiles (via contextily) for a region's data-card lon/lat ROI, warps them
from Web Mercator onto the environment's latitude-linear grid registration
(row 0 = max latitude, matching ``firms_to_cells``), and caches the result both
in-process and on disk so GIF rendering fetches tiles at most once per region.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from wildfire_marl.paths import data_dir

# Data-card ROIs, duplicated from wildfire_marl.data.to_cell2fire.REGIONS to avoid
# importing the (heavy, xarray-dependent) conversion module from the viz layer.
REGION_ROIS: dict[str, tuple[float, float, float, float]] = {
    "saudi": (45.5, 23.5, 50.5, 28.5),
    "california": (-124.5, 36.5, -119.0, 41.5),
}

_memory_cache: dict[tuple[str, int], np.ndarray | None] = {}


def _mercator_y(lat_deg: float) -> float:
    lat = math.radians(lat_deg)
    return math.log(math.tan(math.pi / 4 + lat / 2))


def _warp_to_latlon_grid(
    tiles: np.ndarray,
    extent_3857: tuple[float, float, float, float],
    roi: tuple[float, float, float, float],
    size: int,
) -> np.ndarray:
    """Resample a Web Mercator tile mosaic onto a (size, size) lat/lon-linear image."""
    lon_min, lat_min, lon_max, lat_max = roi
    x_min, x_max, y_min, y_max = extent_3857
    h, w = tiles.shape[:2]

    lons = np.linspace(lon_min, lon_max, size, endpoint=False) + (lon_max - lon_min) / (2 * size)
    lats = np.linspace(lat_max, lat_min, size, endpoint=False) - (lat_max - lat_min) / (2 * size)

    earth_r = 6378137.0
    xs = np.radians(lons) * earth_r
    ys = np.array([_mercator_y(la) for la in lats]) * earth_r

    cols = np.clip(((xs - x_min) / (x_max - x_min) * w).astype(int), 0, w - 1)
    rows = np.clip(((y_max - ys) / (y_max - y_min) * h).astype(int), 0, h - 1)
    return tiles[np.ix_(rows, cols)][..., :3]


def fetch_region_basemap(region: str, size: int = 512) -> np.ndarray | None:
    """Return an RGB (size, size, 3) uint8 OSM basemap for the region ROI.

    Returns None when the region has no ROI on file or tiles cannot be fetched
    (offline); callers fall back to the plain background.
    """
    region = region.lower()
    key = (region, size)
    if key in _memory_cache:
        return _memory_cache[key]

    roi = REGION_ROIS.get(region)
    if roi is None:
        _memory_cache[key] = None
        return None

    cache_path = data_dir() / "basemaps" / f"{region}_osm_{size}.npy"
    if cache_path.exists():
        img = np.load(cache_path)
        _memory_cache[key] = img
        return img

    try:
        import contextily as ctx

        lon_min, lat_min, lon_max, lat_max = roi
        tiles, extent = ctx.bounds2img(
            lon_min, lat_min, lon_max, lat_max, ll=True, source=ctx.providers.OpenStreetMap.Mapnik
        )
        img = _warp_to_latlon_grid(tiles, extent, roi, size).astype(np.uint8)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, img)
    except Exception:
        img = None

    _memory_cache[key] = img
    return img
