# data/

Datasets are **not tracked in git** (size + third-party licenses). This directory holds:

```
data/
├── sample/                      # tiny synthetic tensor, COMMITTED (for tests + quickstart)
│   └── state_tensor.npy
└── <region>/                    # generated locally or downloaded; git-ignored
    ├── raw/{era5,firms,dem,ndvi}/
    ├── processed/...
    └── grids/{32x32,64x64}/{<channel>.npy, state_tensor.npy, state_metadata.json}
```

## Rebuild from source

Acquisition and preprocessing of the four remote-sensing sources run as notebooks, one per
source per region (credentials go in `.env`, see `.env.example`):

```
notebooks/Data Preliminary/<Region>/01_era5_pipeline_<region>.ipynb    # ERA5 weather
notebooks/Data Preliminary/<Region>/02_firms_pipeline_<region>.ipynb   # FIRMS hotspots
notebooks/Data Preliminary/<Region>/03_dem_pipeline_<region>.ipynb     # SRTM terrain
notebooks/Data Preliminary/<Region>/04_ndvi_pipeline_<region>.ipynb    # MODIS NDVI fuel
```

The resulting grids are then converted into Cell2Fire landscapes and infrastructure rasters:

```bash
python -m wildfire_marl.data.to_cell2fire --region saudi_eastern_province --grid 32
python -m wildfire_marl.infra.build_infrastructure --region saudi
```

## Or fetch a prepared bundle

A fixed, citeable snapshot (with sha256 manifest) can be archived on Zenodo. After
downloading, verify with:

```bash
python -m wildfire_marl.reproducibility.make_manifest   # regenerate, or
python -c "from wildfire_marl.reproducibility.manifest import verify_manifest; print(verify_manifest('data','results/data_manifest.json'))"
```

See [`../docs/data_card.md`](../docs/data_card.md) for sources, licenses, CRS, and coverage.
