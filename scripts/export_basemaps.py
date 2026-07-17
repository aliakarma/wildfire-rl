"""Export the GIS basemap behind each region's grid for the interactive replay viewer.

The Phase-5/6 GIFs render the rollout over real basemap tiles via
``wildfire_marl.viz.geo_renderer.GeoRenderer``; the browser cannot fetch tiles (CSP,
offline builds, attribution churn), so this script bakes the *same* basemap — same tile
provider, same extent — into a static image plus the cell geometry needed to register the
32x32 simulation grid onto it.

Geometry is taken from ``GeoRenderer`` itself rather than recomputed, so the viewer and the
GIFs place every cell identically. Latitude is constant per grid row and longitude per grid
column, so the per-cell centres collapse to two 1-D arrays; they are emitted as fractions of
the cropped image (Mercator row spacing varies ~4-7% across a 500 km extent, so a uniform
grid would misregister cells by up to a quarter-cell).

Runs in the WSL venv (needs contextily), like all rendering tooling::

    python scripts/export_basemaps.py
    python scripts/export_basemaps.py --regions saudi --style OpenTopoMap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from wildfire_marl.viz.geo_renderer import _REGION_META, GeoRenderer  # noqa: E402

REGIONS = ["saudi", "california"]
OUT_DEFAULT = REPO / "dashboard" / "public" / "data" / "basemaps"


def crop_to_extent(
    img: np.ndarray,
    ext: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    width: int,
) -> Image.Image:
    """Crop the tile mosaic to exactly ``target`` (Web Mercator) and resample to ``width``.

    ``bounds2img`` returns a tile-aligned mosaic strictly larger than the requested bounds.
    The crop box is fractional, so it is applied through ``Image.resize(box=...)``, which
    resamples at sub-pixel precision — an integer crop would shift the grid registration by
    up to a pixel.
    """
    src = Image.fromarray(np.asarray(img)[:, :, :3].astype("uint8"), "RGB")
    ex0, ex1, ey0, ey1 = ext
    tx0, tx1, ty0, ty1 = target
    sw, sh = src.size
    px_x = sw / (ex1 - ex0)
    px_y = sh / (ey1 - ey0)
    # origin="upper": image row 0 is ey1 (north).
    box = ((tx0 - ex0) * px_x, (ey1 - ty1) * px_y, (tx1 - ex0) * px_x, (ey1 - ty0) * px_y)
    height = max(1, round(width * (ty1 - ty0) / (tx1 - tx0)))
    return src.resize((width, height), Image.LANCZOS, box=box)


def export_region(
    region: str, style: str, data_dir: str, out_dir: Path, width: int, quality: int
) -> dict:
    r = GeoRenderer(region, style=style, data_dir=data_dir)
    img, ext = r._fetch_basemap()
    x0, x1, y0, y1 = r.extent_merc
    out = crop_to_extent(img, ext, r.extent_merc, width)

    name = f"{region}.jpg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.save(out_dir / name, "JPEG", quality=quality, optimize=True, progressive=True)

    span_x, span_y = x1 - x0, y1 - y0
    # Cell centres as fractions of the cropped image: x from the first row, y from the
    # first column (lat is constant per row, lon per column — asserted below).
    assert np.ptp(r._cell_y, axis=1).max() == 0, "latitude varies within a grid row"
    assert np.ptp(r._cell_x, axis=0).max() == 0, "longitude varies within a grid column"
    cx = [(float(v) - x0) / span_x for v in r._cell_x[0, :]]
    cy = [(y1 - float(v)) / span_y for v in r._cell_y[:, 0]]

    meta = _REGION_META[region]
    kb = (out_dir / name).stat().st_size / 1024
    print(
        f"  {region}: {name} ({out.size[0]}x{out.size[1]}, {kb:.0f} KB) "
        f"<- {r.tile_info['name']}"
    )
    return {
        "image": name,
        "width": out.size[0],
        "height": out.size[1],
        "grid": int(r.grid_size),
        "displayName": meta["display_name"],
        "assetLabel": meta["asset_label"],
        "assetColor": meta["asset_color"],
        # Fractions of the image (0-1): centre of each column / row, and the uniform
        # half-cell size GeoRenderer draws with.
        "cellX": [round(v, 6) for v in cx],
        "cellY": [round(v, 6) for v in cy],
        "halfW": round(r._cell_half_w / span_x, 6),
        "halfH": round(r._cell_half_h / span_y, 6),
        "bounds": {
            "lat": [round(r.lat_bounds[0], 6), round(r.lat_bounds[1], 6)],
            "lon": [round(r.lon_bounds[0], 6), round(r.lon_bounds[1], 6)],
        },
        # A 100 km scale bar as a fraction of image width — the GIF draws the same bar.
        "scaleBarFrac": round(100_000 / span_x, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regions", default=",".join(REGIONS))
    ap.add_argument("--style", default="EsriWorldTopo")
    ap.add_argument("--data-dir", default=str(REPO / "data" / "cell2fire"))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--width", type=int, default=1280, help="exported image width (px)")
    ap.add_argument("--quality", type=int, default=82, help="JPEG quality")
    args = ap.parse_args()

    out_dir = Path(args.out)
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    print(f"Exporting basemaps ({args.style}) -> {out_dir}")

    entries = {}
    for region in regions:
        entries[region] = export_region(
            region, args.style, args.data_dir, out_dir, args.width, args.quality
        )

    attribution = GeoRenderer(regions[0], style=args.style, data_dir=args.data_dir).tile_info
    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "style": args.style,
                "tileName": attribution["name"],
                # Attribution is a licence condition of the tile providers — the viewer
                # renders this string over the map, as the GIFs do.
                "attribution": attribution["attribution"],
                "regions": entries,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{len(entries)} basemap(s) + index.json -> {out_dir}")


if __name__ == "__main__":
    main()
