"""One-time basemap tile prefetcher for offline GIS rendering.

Downloads and caches the background map tiles for both regions so that subsequent
``GeoRenderer`` calls work fully offline.  Tiles are cached by contextily in its
default cache directory (``~/.cache/contextily/`` on Linux).

Usage::

    python scripts/prefetch_basemaps.py
    python scripts/prefetch_basemaps.py --styles EsriWorldTopo,OpenTopoMap
    python scripts/prefetch_basemaps.py --regions saudi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from wildfire_marl.viz.geo_renderer import _TILE_PROVIDERS, GeoRenderer


def main() -> None:
    ap = argparse.ArgumentParser(description="Prefetch basemap tiles for offline rendering.")
    ap.add_argument("--regions", default="saudi,california")
    ap.add_argument("--styles", default="EsriWorldTopo")
    ap.add_argument("--data-dir", default="data/cell2fire")
    args = ap.parse_args()

    regions = [r.strip() for r in args.regions.split(",")]
    styles = [s.strip() for s in args.styles.split(",")]

    for style in styles:
        if style not in _TILE_PROVIDERS:
            print(f"Unknown style '{style}'; available: {sorted(_TILE_PROVIDERS)}")
            sys.exit(1)

    for region in regions:
        for style in styles:
            print(f"Fetching tiles: {region} / {style}...", end=" ", flush=True)
            renderer = GeoRenderer(region, style=style, data_dir=args.data_dir)
            try:
                img, ext = renderer._fetch_basemap()
                h, w = img.shape[:2]
                print(f"OK ({w}x{h} px, extent {[round(e) for e in ext]})")
            except Exception as e:
                print(f"FAILED: {e}")

    cache_path = Path.home() / ".cache" / "contextily"
    if cache_path.exists():
        n = sum(1 for _ in cache_path.rglob("*") if _.is_file())
        print(f"\nCached tiles: {n} files in {cache_path}")
    print("Done. Subsequent renders will work offline.")


if __name__ == "__main__":
    main()
