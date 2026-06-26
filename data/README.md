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

```bash
python scripts/download_data.py --region saudi_eastern_province --source all  # needs .env creds
python scripts/build_tensors.py --region saudi_eastern_province --grid 32
```

## Or fetch a prepared bundle

A fixed, citeable snapshot (with sha256 manifest) can be archived on Zenodo. After
downloading, verify with:

```bash
python scripts/make_manifest.py   # regenerate, or
python -c "from wildfire_rl.data.manifest import verify_manifest; print(verify_manifest('data','results/data_manifest.json'))"
```

See [`../docs/data_card.md`](../docs/data_card.md) for sources, licenses, CRS, and coverage.
