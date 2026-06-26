# results/

Small, canonical result tables (CSV/JSON) **are** committed here so the headline numbers
are visible without rerunning. Large/derived outputs are git-ignored:

- `runs/` — per-run reproducibility metadata (git SHA, config hash, seed) — git-ignored.
- `*.parquet` and `logs/` — git-ignored.
- `legacy_notebook_runs/` — the original notebook CSVs, kept for provenance. **Superseded**
  by the canonical pipeline (`wildfire-rl train/evaluate/transfer`), which fixes the metric
  threshold inconsistency and the asymmetric/partly-hardcoded transfer tables.

Regenerate canonical results: `make train && make evaluate && make transfer`.
