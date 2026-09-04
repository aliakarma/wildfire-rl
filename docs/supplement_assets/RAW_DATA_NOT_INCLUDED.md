# Raw geospatial data: what was omitted and how to regenerate it

**Nothing needed to run the experiments is missing.** The simulator's actual inputs —
`data/cell2fire/California/` and `data/cell2fire/Saudi/` — are complete and total ~0.2 MB.
Every experiment in the paper reads those files.

What is omitted is the **upstream provenance chain** that produced them: 703 MB of raw
rasters, which exceeds the 50 MB supplement cap by an order of magnitude.

## What is included

| Path | Contents |
|---|---|
| `data/cell2fire/{California,Saudi}/` | Cell2Fire landscape: `Forest.asc`, `Data.csv`, `Weather.csv`, `Ignitions.csv`, `elevation.asc`, `slope.asc`, `fbp_lookup_table.csv` |
| `data/cell2fire/{California,Saudi}/` | Infrastructure layer: `criticality.asc/.npy`, `asset_type.asc/.npy`, `blast_radius.asc/.npy` |
| `data/cell2fire/{California,Saudi}/` | Provenance: `conversion_report.json`, `ignition_candidates.json` |
| `data/{california,saudi_eastern_province}/grids/` | 7-channel state tensors at 32×32 and 64×64, plus per-channel arrays and `state_metadata.json` |
| `data/sample/` | 2 KB tensor used by the fast smoke test |

## What is omitted, and how to get it back

| Omitted | Size | Regenerate with |
|---|---:|---|
| SRTM 3-arcsec DEM tiles (`*.hgt.gz`) and the stitched `*_dem_3arcsec.npy` | 620 MB | `python scripts/fetch_srtm_dem.py` — downloads from NASA SRTM using the tile list in `data/*/raw/dem/*_dem_manifest.json` |
| ERA5 NetCDF (`era5_*_june_2025.nc`) | 1.1 MB | Copernicus CDS API; variables, area, and time window in `docs/data_card.md`. Needs a free CDS account. |
| FIRMS active-fire CSVs | 0.5 MB | NASA FIRMS archive download; query parameters in `docs/data_card.md` |
| Sentinel-2 NDVI GeoTIFF (`saudi_ndvi.tif`) | 7.4 MB | Sentinel-2 L2A composite; scene ids and date range in `docs/data_card.md` |
| Slope GeoTIFFs | 0.9 MB | Derived from the DEM — recomputed automatically by the conversion pipeline |

To rebuild the landscapes from raw data after refetching:

```bash
python -m wildfire_marl.data.to_cell2fire --region saudi
python -m wildfire_marl.data.to_cell2fire --region california
```

The result should reproduce the shipped `data/cell2fire/*` byte-for-byte; every file's
SHA-256 is recorded in `SHA256SUMS.txt` at the archive root, so this is checkable.

## Why the checksums still work

`conversion_report.json` and `ignition_candidates.json` record the *source* file each
landscape was derived from. In this archive those paths were rewritten from absolute build
paths to repository-relative ones during anonymization — that is the only edit made to any
data file.
