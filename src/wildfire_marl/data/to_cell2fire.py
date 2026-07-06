"""Region tensors → Cell2Fire landscape inputs (Phase 2, REMEDIATION_PLAN_V2).

Converts the preserved V1 geospatial layers (MODIS-NDVI fuel, SRTM slope, ERA5 weather,
FIRMS hotspots) for one region into a Cell2Fire instance folder:

    data/cell2fire/<Region>/
        Forest.asc               FBP fuel-type grid (shared NDVI→FBP mapping, fuel_mapping.py)
        elevation.asc            per-cell mean elevation (m) from the 3-arcsec SRTM mosaic
        slope.asc                slope (degrees) from the V1 SRTM-derived raster (provenance)
        Data.csv                 per-cell FBP inputs (fueltype, real lat/lon, SRTM elev/ps/saz)
        Weather.csv              hourly ERA5 series + daily CFFDRS FWI codes (fwi.py)
        fbp_lookup_table.csv     standard Prometheus/Cell2Fire lookup (copied verbatim)
        Ignitions.csv            deterministic default ignition (top FIRMS cell on fuel)
        ignition_candidates.json FIRMS-derived ignition set (ignition.py; disjoint-seed sampling)
        conversion_report.json   provenance: input hashes, class distribution, checks

Declared modeling abstractions (documented in docs/data_card.md, ablated in Phase 10):
  * **Representative-tile cell size.** Forest.asc declares cellsize=100 m — Cell2Fire's
    validated operating resolution — while the fuel/weather *patterns* come from the regional
    grids (ROI cells are ~17 km). Absolute geographic distance is not preserved; the
    abstraction is identical for both regions, so cross-region comparisons remain internally
    consistent.
  * **Representative-tile topography in Data.csv.** elev/ps/saz come from the 3-arcsec SRTM
    mosaic (scripts/fetch_srtm_dem.py; closes the V1 slope-magnitude-only data debt). Per ROI
    cell: elev = block-mean elevation; ps = block mean of the fine-scale (~90 m) slope
    percent — the steepness at the declared 100 m operating resolution, not the near-zero
    plane fit across a ~17 km cell; saz = azimuth of the block-mean gradient (dominant
    UPHILL direction — the FBP convention: fbp.c pushes the slope-equivalent wind along saz
    exactly like waz). The V1 slope.asc still ships as a cross-check (correlation reported
    in conversion_report.json).
  * **Zero precipitation.** The archived ERA5 extracts carry no precipitation variable; both
    ROIs are June-dry. APCP=0 with FWI start-up defaults.

The converter is deterministic: identical inputs produce byte-identical outputs.

    python -m wildfire_marl.data.to_cell2fire --region saudi --grid 32
    python -m wildfire_marl.data.to_cell2fire --region california --grid 32
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wildfire_marl.data.fuel_mapping import (
    GRASS_CURING,
    GRASS_FUEL_LOAD,
    NONFUEL_CODE,
    class_distribution,
    code_to_fueltype,
    fuel_channel_to_ndvi,
    ndvi_to_fbp_codes,
)
from wildfire_marl.data.fwi import FWIState, daily_fwi
from wildfire_marl.data.ignition import build_candidates, write_ignitions_csv
from wildfire_marl.paths import data_dir, repo_root
from wildfire_marl.reproducibility.logging_utils import file_sha256

_GRASS_CODES = (31, 32)
_CELLSIZE_M = 100  # representative-tile abstraction — see module docstring


@dataclass(frozen=True)
class RegionSpec:
    name: str  # canonical short name used on the CLI
    out_name: str  # folder name under data/cell2fire/
    dir: str  # folder under data/
    roi: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max
    era5: str
    firms: str
    slope_tif: str
    dem_npy: str  # 3-arcsec SRTM ROI mosaic built by scripts/fetch_srtm_dem.py
    ndvi_tif: str | None  # raw NDVI for the recovery cross-check (Saudi only)


REGIONS: dict[str, RegionSpec] = {
    "saudi": RegionSpec(
        name="saudi",
        out_name="Saudi",
        dir="saudi_eastern_province",
        roi=(45.5, 23.5, 50.5, 28.5),
        era5="raw/era5/era5_saudi_june_2025.nc",
        firms="raw/firms/active_fire/saudi_firms_june_2025.csv",
        slope_tif="raw/dem/elevation/saudi_slope.tif",
        dem_npy="raw/dem/srtm/saudi_dem_3arcsec.npy",
        ndvi_tif="raw/ndvi/vegetation/saudi_ndvi.tif",
    ),
    "california": RegionSpec(
        name="california",
        out_name="California",
        dir="california",
        roi=(-124.5, 36.5, -119.0, 41.5),
        era5="raw/era5/era5_california_june_2025.nc",
        firms="raw/firms/active_fire/california_firms_june_2025.csv",
        slope_tif="raw/dem/california_slope.tif",
        dem_npy="raw/dem/srtm/california_dem_3arcsec.npy",
        ndvi_tif=None,  # raw NDVI not archived by V1 — recorded data-debt (see fuel_mapping)
    ),
}


# ------------------------------------------------------------------ small helpers
def _block_mean(a: np.ndarray, grid: int) -> np.ndarray:
    """Deterministic block-mean resample of a 2D array to (grid, grid)."""
    h, w = a.shape
    hh, ww = (h // grid) * grid, (w // grid) * grid
    a = a[:hh, :ww].reshape(grid, hh // grid, grid, ww // grid)
    return a.mean(axis=(1, 3))


def _write_asc(path: Path, grid_values: np.ndarray, fmt: str) -> None:
    h, w = grid_values.shape
    lines = [
        f"ncols {w}",
        f"nrows {h}",
        "xllcorner 0",
        "yllcorner 0",
        f"cellsize {_CELLSIZE_M}",
        "NODATA_value -9999",
    ]
    for row in grid_values:
        lines.append(" ".join(fmt % v for v in row))
    path.write_text("\n".join(lines) + "\n")


_M_PER_DEG = 111_320.0  # WGS84 mean metres per degree of latitude


def _block_reduce(a: np.ndarray, grid: int) -> np.ndarray:
    """Mean of `a` over (grid, grid) cells with edge-exact (rounded linspace) bins.

    Unlike ``_block_mean`` this does not truncate the remainder rows/cols, so the cell
    footprints stay registered to the ROI edges (the DEM mosaic is edge-exact).
    """
    h, w = a.shape
    re = np.rint(np.linspace(0, h, grid + 1)).astype(int)
    ce = np.rint(np.linspace(0, w, grid + 1)).astype(int)
    out = np.empty((grid, grid), dtype=np.float64)
    for i in range(grid):
        for j in range(grid):
            out[i, j] = a[re[i] : re[i + 1], ce[j] : ce[j + 1]].mean()
    return out


def _topography(
    spec: RegionSpec, region_root: Path, grid: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell (elev m, ps slope-%, saz uphill-azimuth °) from the 3-arcsec SRTM mosaic.

    The mosaic (scripts/fetch_srtm_dem.py) is north-up and edge-exact on the ROI, so its
    pixels bin cleanly into the row-0-north grid cells. Per cell:
      * elev — block-mean elevation;
      * ps   — block mean of the fine-scale (~90 m) slope magnitude in percent: the
        steepness at the declared 100 m cell size (same semantics as V1's slope.asc),
        NOT the near-zero plane fit across a ~17 km ROI cell;
      * saz  — compass azimuth of the block-mean gradient vector, i.e. the dominant
        UPHILL direction. FBP convention: slope_effect (FBPfunc5_NoDebug.c) adds the
        slope-equivalent wind along saz exactly as waz, and fires run upslope.
    """
    dem_path = region_root / spec.dem_npy
    if not dem_path.exists():
        raise FileNotFoundError(
            f"{dem_path} missing — run: python scripts/fetch_srtm_dem.py --region {spec.name}"
        )
    z = np.load(dem_path).astype(np.float64)
    lon_min, lat_min, lon_max, lat_max = spec.roi
    h, w = z.shape
    lat_px = lat_max - (np.arange(h) + 0.5) * (lat_max - lat_min) / h
    dy_m = (lat_max - lat_min) / h * _M_PER_DEG
    dx_m = (lon_max - lon_min) / w * _M_PER_DEG * np.cos(np.radians(lat_px))
    dz_drow, dz_dcol = np.gradient(z)
    dz_dnorth = -dz_drow / dy_m  # row index increases southward
    dz_deast = dz_dcol / dx_m[:, None]

    elev = _block_reduce(z, grid)
    ps = _block_reduce(100.0 * np.hypot(dz_deast, dz_dnorth), grid)
    g_e = _block_reduce(dz_deast, grid)
    g_n = _block_reduce(dz_dnorth, grid)
    saz = (np.degrees(np.arctan2(g_e, g_n)) + 360.0) % 360.0
    return elev, ps, saz


def _relative_humidity(t2m_k: np.ndarray, d2m_k: np.ndarray) -> np.ndarray:
    """RH (%) from temperature/dewpoint (K), Magnus formula (Alduchov & Eskridge 1996)."""
    t = t2m_k - 273.15
    td = d2m_k - 273.15
    rh = 100.0 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
    return np.clip(rh, 0.0, 100.0)


def _wind_from_azimuth_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological wind direction: compass bearing the wind blows FROM (0=N, 90=E).

    Cell2Fire's weather reader converts this to the spread-push azimuth itself
    (``ReadCSV.cpp: waz = WD + 180``), so ``Weather.csv`` must carry the FROM-direction.
    The convention is verified empirically by the sim-smoke drift check (fire centroid
    drift must align with WD+180); the initial Phase-2 run caught exactly the 180° error
    when this column carried the TOWARD-direction instead.
    """
    return (np.degrees(np.arctan2(u, v)) + 180.0) % 360.0


# ------------------------------------------------------------------ weather
def _build_weather_rows(spec: RegionSpec, region_root: Path) -> tuple[list[dict], dict]:
    """Hourly ERA5 spatial-mean series + daily CFFDRS FWI codes (held within each day).

    The archived ERA5 extracts are 6-hourly; Cell2Fire consumes one weather row per
    ``MinutesPerWP`` (default 60) simulated minutes, i.e. rows are hourly. The u/v/t/dewpoint
    fields are therefore linearly interpolated from 6-hourly to hourly (standard practice for
    slowly-varying screen-level analyses), and RH/WS/WD are recomputed from the interpolated
    fields. Note the Cell2Fire time semantics this implies for the env: one fire period =
    ``Fire-Period-Length`` = 1 simulated minute; an agent step advances ``steps_per_action``
    fire periods (so ``steps_per_action=60`` = one action per simulated hour).
    """
    from netCDF4 import Dataset, num2date  # optional geo dependency

    ds = Dataset(region_root / spec.era5)
    try:
        times = ds.variables["valid_time"]
        dts6 = [
            datetime(*t.timetuple()[:6], tzinfo=timezone.utc)
            for t in num2date(times[:], times.units)
        ]
        sm6 = {
            v: np.asarray(ds.variables[v][:]).mean(axis=(1, 2))
            for v in ("u10", "v10", "t2m", "d2m")
        }
    finally:
        ds.close()

    # 6-hourly -> hourly linear interpolation on the epoch-seconds axis.
    t6 = np.array([d.timestamp() for d in dts6])
    t1 = np.arange(t6[0], t6[-1] + 1, 3600.0)
    dts = [datetime.fromtimestamp(s, tz=timezone.utc) for s in t1]
    sm = {v: np.interp(t1, t6, sm6[v]) for v in sm6}

    tmp_c = sm["t2m"] - 273.15
    rh = _relative_humidity(sm["t2m"], sm["d2m"])
    ws_kmh = np.hypot(sm["u10"], sm["v10"]) * 3.6
    wd = _wind_from_azimuth_deg(sm["u10"], sm["v10"])

    # Daily FWI over calendar days (spatial-mean daily means; APCP=0 — see module docstring).
    day_keys = sorted({d.date() for d in dts})
    state = FWIState()
    day_codes: dict = {}
    for day in day_keys:
        idx = [i for i, d in enumerate(dts) if d.date() == day]
        state, codes = daily_fwi(
            temp=float(np.mean(tmp_c[idx])),
            rh=float(np.mean(rh[idx])),
            wind=float(np.mean(ws_kmh[idx])),
            rain=0.0,
            month=day.month,
            prev=state,
        )
        day_codes[day] = codes

    # Cell2Fire (fork AND upstream) holds weather in a fixed 150-slot buffer
    # (Cell2Fire.cpp:40 `weatherDF wdf[150]`); more rows corrupt the heap. We therefore emit
    # only the FINAL 144 hourly rows (6 days, June 25-30) — the most fire-mature window —
    # while the FWI codes retain the full-month spin-up computed above. 144 rows bound an
    # episode at ~143 agent-steps of 60 sim-minutes each.
    keep = 144
    rows = []
    for i, d in enumerate(dts):
        c = day_codes[d.date()]
        rows.append(
            {
                "Scenario": spec.out_name,
                "datetime": d.strftime("%Y-%m-%d %H:%M"),
                "APCP": 0.0,
                "TMP": round(float(tmp_c[i]), 1),
                "RH": int(round(float(rh[i]))),
                "WS": round(float(ws_kmh[i]), 1),
                "WD": int(round(float(wd[i]))),
                "FFMC": c["ffmc"],
                "DMC": c["dmc"],
                "DC": c["dc"],
                "ISI": c["isi"],
                "BUI": c["bui"],
                "FWI": c["fwi"],
            }
        )
    rows = rows[-keep:]
    summary = {
        "n_rows": len(rows),
        "window": [rows[0]["datetime"], rows[-1]["datetime"]],
        "fwi_spinup_days": [str(day_keys[0]), str(day_keys[-1])],
        "mean_tmp_c": round(float(tmp_c.mean()), 2),
        "mean_rh_pct": round(float(rh.mean()), 2),
        "mean_ws_kmh": round(float(ws_kmh.mean()), 2),
        "final_day_codes": day_codes[day_keys[-1]],
    }
    return rows, summary


# ------------------------------------------------------------------ main conversion
def convert(region: str, grid: int, out_root: Path | None = None) -> Path:
    spec = REGIONS[region]
    region_root = data_dir() / spec.dir
    out = (out_root or (data_dir() / "cell2fire")) / spec.out_name
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {"region": region, "grid": grid, "cellsize_m_declared": _CELLSIZE_M}

    # ---- 1. fuel: committed V1 fuel channel -> physical NDVI -> shared FBP mapping --------
    fuel = np.load(region_root / "grids" / f"{grid}x{grid}" / "fuel.npy")
    ndvi = fuel_channel_to_ndvi(region, fuel)
    codes = ndvi_to_fbp_codes(ndvi)
    report["ndvi"] = {
        "min": round(float(ndvi.min()), 4),
        "max": round(float(ndvi.max()), 4),
        "mean": round(float(ndvi.mean()), 4),
    }
    report["fuel_class_distribution"] = class_distribution(codes)

    # ---- 2. Saudi-only cross-check: recovered NDVI vs raw MODIS tif -----------------------
    if spec.ndvi_tif is not None:
        try:
            import tifffile

            raw = np.asarray(tifffile.imread(region_root / spec.ndvi_tif), dtype=np.float64)
            raw_ndvi = _block_mean(raw, grid) * 1e-4  # MODIS scale factor
            r = float(np.corrcoef(raw_ndvi.ravel(), ndvi.ravel())[0, 1])
            mae = float(np.mean(np.abs(raw_ndvi - ndvi)))
            report["ndvi_recovery_crosscheck"] = {
                "pearson_r_vs_raw_modis": round(r, 4),
                "mae": round(mae, 4),
                "raw_mean": round(float(raw_ndvi.mean()), 4),
                "recovered_mean": round(float(ndvi.mean()), 4),
            }
        except ImportError:
            report["ndvi_recovery_crosscheck"] = "skipped (tifffile not installed)"

    # ---- 3. topography: SRTM-derived elev/ps/saz + elevation.asc ---------------------------
    elev, ps, saz = _topography(spec, region_root, grid)
    _write_asc(out / "elevation.asc", elev, "%.1f")
    # Integer per-cell values — Cell2Fire parses elev/ps/saz with std::stoi (ReadCSV.cpp).
    elev_i = np.rint(elev).astype(int)
    ps_i = np.rint(ps).astype(int)
    saz_i = np.rint(saz).astype(int) % 360
    saz_i[ps_i == 0] = 0  # azimuth is meaningless (and inert: wse=0) on flat cells
    report["topography"] = {
        "dem_npy": spec.dem_npy,
        "elev_m": {
            "min": int(elev_i.min()),
            "max": int(elev_i.max()),
            "mean": round(float(elev.mean()), 1),
        },
        "slope_pct": {"mean": round(float(ps.mean()), 2), "max": round(float(ps.max()), 2)},
        "cells_with_slope_effect": int((ps_i > 0).sum()),
        "saz_convention": "uphill azimuth (FBP; = downslope aspect + 180)",
    }

    # ---- 4. Forest.asc + slope.asc (V1 raster kept as provenance + cross-check) -----------
    _write_asc(out / "Forest.asc", codes, "%d")
    try:
        import tifffile

        slope_deg = _block_mean(
            np.asarray(tifffile.imread(region_root / spec.slope_tif), dtype=np.float64), grid
        )
        _write_asc(out / "slope.asc", slope_deg, "%.2f")
        # Same quantity at different provenance/resolution: V1 slope (deg) as percent vs
        # the new SRTM-derived ps — correlation guards against a mis-registered mosaic.
        v1_pct = 100.0 * np.tan(np.radians(slope_deg))
        report["slope_deg"] = {
            "mean": round(float(slope_deg.mean()), 3),
            "max": round(float(slope_deg.max()), 3),
            "pearson_r_v1_vs_srtm_ps": round(
                float(np.corrcoef(v1_pct.ravel(), ps.ravel())[0, 1]), 4
            ),
        }
    except ImportError:
        report["slope_deg"] = "skipped (tifffile not installed)"

    # ---- 5. Data.csv (per-cell FBP inputs; SRTM topography — see module docstring) --------
    lon_min, lat_min, lon_max, lat_max = spec.roi
    header = (
        "fueltype,mon,jd,M,jd_min,lat,lon,elev,ffmc,ws,waz,bui,ps,saz,pc,pdf,gfl,cur,time,pattern"
    )
    lines = [header]
    for r_i in range(grid):
        lat = lat_max - (r_i + 0.5) * (lat_max - lat_min) / grid
        for c_i in range(grid):
            lon = lon_min + (c_i + 0.5) * (lon_max - lon_min) / grid
            code = int(codes[r_i, c_i])
            ft = code_to_fueltype(code)
            gfl = f"{GRASS_FUEL_LOAD}" if code in _GRASS_CODES else ""
            cur = f"{GRASS_CURING}" if code in _GRASS_CODES else ""
            lines.append(
                f"{ft},,,,,{lat:.6f},{lon:.6f},{elev_i[r_i, c_i]},,,,,"
                f"{ps_i[r_i, c_i]},{saz_i[r_i, c_i]},,,{gfl},{cur},20,"
            )
    (out / "Data.csv").write_text("\n".join(lines) + "\n")

    # ---- 6. Weather.csv ---------------------------------------------------------------------
    rows, weather_summary = _build_weather_rows(spec, region_root)
    wcols = list(rows[0].keys())
    wlines = [",".join(wcols)]
    for row in rows:
        wlines.append(",".join(str(row[c]) for c in wcols))
    (out / "Weather.csv").write_text("\n".join(wlines) + "\n")
    report["weather"] = weather_summary

    # ---- 7. standard FBP lookup table (verbatim copy) --------------------------------------
    lookup_src = (
        repo_root() / "third_party" / "firehose" / "data" / "Sub40x40" / "fbp_lookup_table.csv"
    )
    shutil.copyfile(lookup_src, out / "fbp_lookup_table.csv")

    # ---- 8. FIRMS ignition candidates + deterministic default ignition ---------------------
    candidates = build_candidates(region_root / spec.firms, spec.roi, grid, codes, NONFUEL_CODE)
    (out / "ignition_candidates.json").write_text(json.dumps(candidates, indent=2))
    if candidates["cells"]:
        # Default ignition: most-detected burnable cell (ties -> lowest cell id).
        w = np.asarray(candidates["weights"])
        default_cell = int(candidates["cells"][int(np.argmax(w))])
    else:
        raise RuntimeError(f"No burnable FIRMS ignition cells for {region}")
    write_ignitions_csv(out / "Ignitions.csv", default_cell)
    report["ignition"] = {
        "default_cell": default_cell,
        "n_candidate_cells": len(candidates["cells"]),
        "n_detections_total": candidates["n_detections_total"],
        "n_detections_on_nonfuel_excluded": candidates["n_detections_on_nonfuel_excluded"],
    }

    # ---- 9. provenance report (deterministic: input hashes, no timestamps) ------------------
    report["inputs_sha256"] = {
        "fuel_npy": file_sha256(region_root / "grids" / f"{grid}x{grid}" / "fuel.npy"),
        "era5_nc": file_sha256(region_root / spec.era5),
        "firms_csv": file_sha256(region_root / spec.firms),
        "slope_tif": file_sha256(region_root / spec.slope_tif),
        "dem_npy": file_sha256(region_root / spec.dem_npy),
        "raw_ndvi_tif": file_sha256(region_root / spec.ndvi_tif) if spec.ndvi_tif else None,
    }
    (out / "conversion_report.json").write_text(json.dumps(report, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True, choices=sorted(REGIONS))
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--out", default=None, help="Output root (default: data/cell2fire)")
    args = ap.parse_args()

    out = convert(args.region, args.grid, Path(args.out) if args.out else None)
    report = json.loads((out / "conversion_report.json").read_text())
    print(f"Converted {args.region} -> {out}")
    print("  fuel classes:", report["fuel_class_distribution"])
    print("  topography:", report["topography"]["elev_m"], report["topography"]["slope_pct"])
    if "ndvi_recovery_crosscheck" in report:
        print("  NDVI cross-check:", report["ndvi_recovery_crosscheck"])
    print(
        "  weather:",
        {
            k: report["weather"][k]
            for k in ("n_rows", "window", "mean_tmp_c", "mean_rh_pct", "mean_ws_kmh")
        },
    )
    print("  ignition:", report["ignition"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
