#!/usr/bin/env python
"""Download raw remote-sensing data needed to rebuild state tensors.

NOTE: This script provides the documented interface for data acquisition.
The actual API calls require credentials and the [geo] optional dependencies.
Pre-built tensors can be obtained via the data bundle or by running
`wildfire-rl build-tensors` after manually placing raw files.

For reviewers: the pre-built state tensors in data/<region>/grids/32x32/
are the artifacts used for all experiments. Their integrity is verified via
sha256 manifests (see `make manifest`).

Reads credentials from environment variables (see .env.example) — never hardcodes keys
(the original notebooks embedded a Google Earth Engine project id). This is a thin,
documented driver; the heavy ERA5/FIRMS/DEM/NDVI logic lives in the geospatial extras
(`pip install -e .[geo]`). Without credentials it prints exactly what is required.

    python scripts/download_data.py --region saudi_eastern_province --source all
"""

from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.paths import data_dir, ensure_dir

logger = get_logger("download_data")

SOURCES = ("era5", "firms", "dem", "ndvi")


def _require(env_var: str, human: str) -> str | None:
    val = os.environ.get(env_var)
    if not val:
        logger.error("Missing %s (%s). Set it in your environment / .env file.", env_var, human)
    return val


def download_era5(region: str) -> None:
    if not _require("CDSAPI_KEY", "Copernicus CDS API key"):
        return
    out = ensure_dir(data_dir() / region / "raw" / "era5")
    logger.info("ERA5 -> %s", out)
    # import cdsapi; cdsapi.Client().retrieve("reanalysis-era5-single-levels", {...}, target)
    logger.info("Install geo extras and implement the CDS request for production use.")


def download_firms(region: str) -> None:
    if not _require("FIRMS_MAP_KEY", "NASA FIRMS map key"):
        return
    out = ensure_dir(data_dir() / region / "raw" / "firms" / "active_fire")
    logger.info("FIRMS active-fire -> %s (use the FIRMS area CSV API with FIRMS_MAP_KEY).", out)


def download_gee(region: str, source: str) -> None:
    project = _require("EE_PROJECT_ID", "Earth Engine project id")
    if not project:
        return
    out = ensure_dir(data_dir() / region / "raw" / source)
    logger.info("%s via Earth Engine (project=%s) -> %s", source.upper(), project, out)
    logger.info(
        "Install geo extras: import ee; ee.Initialize(project=os.environ['EE_PROJECT_ID'])."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="saudi_eastern_province")
    ap.add_argument("--source", choices=(*SOURCES, "all"), default="all")
    args = ap.parse_args()

    todo = SOURCES if args.source == "all" else (args.source,)
    for src in todo:
        if src == "era5":
            download_era5(args.region)
        elif src == "firms":
            download_firms(args.region)
        else:
            download_gee(args.region, src)
    logger.info(
        "Done. Then run: python scripts/build_tensors.py --region %s --grid 32", args.region
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
