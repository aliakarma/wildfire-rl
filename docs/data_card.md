# Data Card

## Summary

Wildfire-RL builds a 7-channel geospatial **state tensor** per region from four public
remote-sensing sources. Raw data is **not redistributed** in this repository; it is
downloaded from the providers (see `scripts/download_data.py`) and processed into tensors.

## Regions

| Region | Directory | ROI (lon/lat rectangle) | Regime |
|---|---|---|---|
| Saudi Eastern Province | `saudi_eastern_province` | `[45.5, 23.5, 50.5, 28.5]` | desert |
| Northern California | `california` | `[-124.5, 36.5, -119.0, 41.5]` | forest/mountain |

Temporal coverage: **June 2025** (single month — a documented limitation for seasonality).

## Channels & sources

| Channel | Source | Dataset | License / policy |
|---|---|---|---|
| fire | NASA FIRMS | active fire (VIIRS/MODIS) | NASA data use policy; requires FIRMS map key |
| fuel | MODIS | NDVI (e.g. MOD13Q1, 250 m) → fuel density | NASA LP DAAC data policy |
| wind_x, wind_y | ECMWF | ERA5 10 m u/v components | Copernicus license (attribution) |
| temperature | ECMWF | ERA5 2 m temperature | Copernicus license |
| humidity | ECMWF | ERA5 2 m dewpoint → humidity proxy | Copernicus license |
| terrain | NASA/USGS | SRTM DEM → slope | public domain (US Gov) |

> ⚠️ A known ERA5 quirk: the time coordinate is `valid_time` (not `time`). The preprocessing
> handles this; see `scripts/download_data.py`.

## Processing

1. Download raw per source → `data/<region>/raw/...`.
2. Resample/clip to the ROI grid; min-max normalize **per channel** to [0, 1]
   (`wildfire_rl.data.normalize.minmax_normalize`).
3. Stack into `(7, G, G)` (`G ∈ {32, 64}`) → `data/<region>/grids/GxG/state_tensor.npy`
   with `state_metadata.json`.

## Normalization caveat (important for transfer)

Per-region min-max means a value of 0.5 in Saudi ≠ 0.5 in California in absolute terms.
For cross-region transfer, use a **shared scaler** (`data/normalize.fit_shared_minmax`) so
absolute climate/terrain differences are not normalized away.

## Intended use / limitations

- Intended for **research** on wildfire-suppression RL and domain transfer — not operational
  firefighting decisions.
- Single month, two regions, simplified fire physics (no combustion chemistry, ember
  transport, or atmospheric coupling). See the report in `docs/paper/`.

## Provenance & integrity

`make manifest` writes sha256 checksums for every artifact under `data/`. A prepared data
bundle may be archived on Zenodo with a DOI for citeable, fixed snapshots.
