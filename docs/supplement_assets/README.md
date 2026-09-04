# Code and Data Supplement

**Infrastructure-Aware Wildfire Coordination: A Benchmark on Validated Fire-Spread Physics
and a Value-Aware Hierarchical Multi-Agent Baseline**

Anonymous submission — AAAI-27.

This archive is **self-contained**. It contains the full source code, the simulator sources,
the simulator's input data, every frozen result the paper cites, the seed-42 model
checkpoints, and a prebuilt interactive dashboard that opens offline in a browser. There are
no links to external code or data repositories.

---

## Verify in 60 seconds — no installation

The dashboard is prebuilt. Serve it and open it:

```bash
python -m http.server 8000 --directory dashboard/dist
# then open http://localhost:8000
```

Every headline number, chart, ablation, transfer matrix, and episode replay in the paper is
there, offline, with the frozen data embedded.

> **It must be served over HTTP.** Opening `dashboard/dist/index.html` directly as a `file://`
> URL will show an empty page — the app fetches its JSON at runtime and browsers block that
> for local files. Any static server works; the one-liner above uses only the Python standard
> library.

## Verify the paper's numbers — ~2 minutes

```bash
pip install -r requirements.txt          # numpy, pandas, scipy, matplotlib
python scripts/verify_supplement.py
```

This re-derives every number in the paper's tables from the frozen **per-episode** CSVs and
diffs them against the values printed in `docs/RESULTS_FROZEN.md` — main results, significance
tests, ablations, the difficulty sweep, and the transfer matrix — then checks every shipped
file against its recorded SHA-256. No GPU, no simulator build, no training.

Expected final line:

```
42 passed, 0 failed, 0 skipped
VERIFICATION PASSED — every claim checked re-derives from the frozen artifacts.
```

## Run the tests

```bash
pip install -e .        # installs the wildfire_marl package itself
pytest tests/
```

Expected on Linux, without the simulator built: **61 passed, 7 skipped** in about 6 seconds.
The 7 skips are the tests that need the compiled Cell2Fire binary; they run once you build it.

## Full reproduction

Build the simulator, then re-evaluate the shipped checkpoints, then optionally retrain:

1. `third_party/firehose/cell2fire/Cell2FireC/BUILD.md` — build Cell2Fire (`make`, needs g++,
   Boost, Eigen3).
2. `docs/REPRODUCIBILITY.md` — the claim → command → expected-number table, in four tiers of
   increasing cost, with honest runtimes.

**Platform: Linux only** (Ubuntu 22.04 / WSL2 / Colab). The simulator is a C++ build and the
native-Windows path is unsupported. Python 3.10+; the frozen runs used 3.12.3.

---

## Repository map

```
README.md                      this file
docs/REPRODUCIBILITY.md        claim -> command -> expected number (start here for repro)
docs/RESULTS_FROZEN.md         the canonical numbers the paper cites, and the freeze audit
docs/data_card.md              dataset provenance, licenses, preprocessing
docs/infra_card.md             critical-infrastructure layer provenance
docs/MIGRATION.md              project history; why no earlier-prototype result is cited

src/wildfire_marl/
  env/cell2fire_binding.py     interactive binding to the Cell2Fire subprocess
  env/marl_env.py              PettingZoo ParallelEnv, N agents, partial observability
  env/rewards.py               WEL/ISR value-weighted reward
  env/regimes.py               difficulty regimes (easy / default / hard)
  agents/heuristics.py         information-matched baselines (No-Op, Value-First, Greedy-Risk, Local Reactive)
  agents/agent_networks.py     MAPPO / CommNet / HierComm policy and value networks
  agents/strategic_controller.py   the hierarchical commander (value-aware rule + learned variant)
  train/                       training drivers for each method
  eval/                        metrics, bootstrap CIs, significance, transfer statistics
  infra/                       critical-asset layer and cascade model
  data/                        region tensors -> Cell2Fire landscapes
  reproducibility/             seeding, manifests, integrity guards
  viz/                         rollout / transfer / comparison rendering

scripts/
  verify_supplement.py         Tier-1 verification (the 2-minute check above)
  run_phase3.py                main results: multi-seed train + eval
  run_phase4_ablations.py      ablations and the difficulty sweep
  run_phase6_transfer.py       Saudi <-> California transfer and generalization
  freeze_results.py            writes MANIFEST.sha256 + FREEZE.json for a results dir
  build_dashboard_data.py      frozen results -> dashboard JSON (fingerprint-gated)
  check_dashboard_consistency.py   asserts dashboard numbers == paper numbers
  fetch_srtm_dem.py            re-downloads the omitted raw elevation data
  render_phase5_gifs.py, render_phase6_transfer.py   regenerate media

data/
  cell2fire/{California,Saudi}/    the simulator's actual inputs (complete, ~0.2 MB)
  {california,saudi_eastern_province}/grids/   7-channel state tensors (32x32, 64x64)
  RAW_DATA_NOT_INCLUDED.md         what was omitted and how to refetch it

results/wildfire_phase3_multiseed/     main results: per-episode CSV, summary, LaTeX table, curves,
                               logs, MANIFEST.sha256, FREEZE.json + seed-42 checkpoints
results/wildfire_phase3_hiercomm_learned/  learned-commander ablation (same shape)
results/wildfire_phase4/               ablations
results/wildfire_phase4_extended/      extended difficulty sweep (supersedes phase4's sweep)
results/wildfire_phase6/               transfer + generalization

third_party/firehose/          Cell2Fire simulator sources (GPL-3.0) + stock benchmark maps
dashboard/                     prebuilt site in dist/, source in src/ and public/
SHA256SUMS.txt                 checksum over every file in this archive
```

The directory names above are **identical to the development tree**, so every command quoted
in `docs/RESULTS_FROZEN.md` and `docs/REPRODUCIBILITY.md` runs verbatim inside this archive.

## Claim → artifact index

The full table is in `docs/REPRODUCIBILITY.md`. In brief:

| Paper claim | Frozen artifact |
|---|---|
| Main results table (WEL / ISR, 7 policies × 2 regions) | `results/wildfire_phase3_multiseed/phase3_summary.json` |
| Per-episode records behind every mean | `results/wildfire_phase3_multiseed/phase3_raw.csv` |
| Significance tests | recomputed by `scripts/verify_supplement.py` |
| Learned-commander ablation | `results/wildfire_phase3_hiercomm_learned/phase3_summary.json` |
| Ablations | `results/wildfire_phase4/ablation_summary.json` |
| Difficulty sweep | `results/wildfire_phase4_extended/robustness_results.csv` |
| Transfer matrix, TRS, asymmetry | `results/wildfire_phase6/transfer_summary.json` |

## What is **not** included, and why

Everything below was omitted to fit the 50 MB cap. Each is regenerable, and nothing needed to
run an experiment is missing.

| Omitted | Size | How to regenerate |
|---|---:|---|
| Raw geospatial rasters (SRTM DEM tiles, NDVI, ERA5, FIRMS) | 703 MB | `python scripts/fetch_srtm_dem.py` and `data/RAW_DATA_NOT_INCLUDED.md` |
| Checkpoints for seeds 1042 / 2042 / 3042 / 4042 | 87 MB | retrain (Tier 4) — **all five seeds' results are in the CSVs**, only the weights are dropped |
| Rendered rollout GIFs | 1.3 GB | `scripts/render_phase5_gifs.py`, `scripts/render_phase6_transfer.py` |
| Compiled simulator (binary, `*.o`, 194 MB precompiled header) | 197 MB | `make` — see `BUILD.md` |
| `node_modules` for the dashboard | 173 MB | `npm ci` — not needed; `dist/` is prebuilt |

The dashboard videos were re-encoded (H.264, width capped at 960 px) and posters converted to
JPEG to fit the cap. The source GIFs are the archival artifact and are regenerable as above.

## Verifying archive integrity

```bash
sha256sum -c SHA256SUMS.txt      # every file in this archive
```

Each `wildfire_phase*/` directory additionally carries its own `MANIFEST.sha256` and a
`FREEZE.json` recording when the results were frozen. `scripts/verify_supplement.py` checks
every shipped file against those recorded hashes. Note that the whole-directory
`fingerprint_sha256` in `FREEZE.json` covers the checkpoints that were dropped under the size
cap, so it cannot be recomputed from this archive — the per-file hashes can, and are.

## License

Our code is MIT (`LICENSE`). The vendored Cell2Fire simulator is GPL-3.0. Data-source terms
and full attribution: `THIRD_PARTY_LICENSES.md`.
