# wildfire-marl — V2 Execution Log (`Phases.md`)

This file records the phase-by-phase execution of `REMEDIATION_PLAN_V2.md` (the Cell2Fire
re-founding). One section is appended per completed phase: exact actions taken, deviations from
the plan (with justification), verification evidence, and the success-gate result. The V1
execution log was retired together with the V1 surface it described (frozen under `legacy_v1/`).

**Repository:** `wildfire-rl` · **Branch:** `v2-cell2fire` · **Executor:** automated (supervised)

---

## Phase 0 — Cleanup, Archival & Migration Preparation

**Status:** ✅ COMPLETE · **Date:** 2026-07-04 · **Branch:** `v2-cell2fire` (created from `additional-branch`)

### Objective (from the plan)

Freeze V1 as a citable historical artifact, carve out the reusable assets into a clean new
package structure, retire the toy simulator, and stand up a Linux development environment ready
for Cell2Fire.

### Actions taken

1. **Branch & freeze.** Created `v2-cell2fire`. Moved the entire V1 custom-sim surface into
   `legacy_v1/` via `git mv` (history preserved — all moves registered as renames):
   - `src/wildfire_rl/` (full package: envs, coordination, routing, ablation, eval, viz, train,
     models, data, cli) → `legacy_v1/src/wildfire_rl/`
   - `scripts/`, `tests/`, `results/`, `figures/`, `configs/`, `notebooks/`, `experimental/`
     → `legacy_v1/<same>/`
   - V1 root surface → `legacy_v1/`: `Makefile`, `Dockerfile`, `.dockerignore`,
     `Colab_Authoritative_Training.ipynb`, `Colab_MARL_Training.ipynb`, `environment.yml`,
     `requirements*.txt` (4 files)
   - V1 docs → `legacy_v1/docs/`: `architecture.md`, `development.md`, `model_card.md`,
     `reproducibility.md`, `REPOSITORY_AUDIT_AND_PLAN.md`, `paper/`
   - `models/` (~5.7 GB V1 checkpoints, git-ignored) → `legacy_v1/models/` (disk rename;
     `.gitignore` updated so it stays untracked)
   - Wrote `legacy_v1/FROZEN.md`: freeze rules (never cited as a result; never edited; never
     imported by V2 code) + the three AAAI rejection reasons. `.pre-commit-config.yaml` gained
     `exclude: ^legacy_v1/` so hooks can never rewrite the archive; ruff/black exclude it too.

2. **Linux environment (WSL2 Ubuntu 24.04).** Native-Windows development abandoned for the sim,
   per the plan. Installed `cmake`, `libeigen3-dev`, `python3-venv`, `python3-pip`
   (gcc/g++/make already present). Verified a trivial Eigen C++ program compiles and runs.

3. **New package skeleton.** `src/wildfire_marl/` with subpackages `env/`, `agents/`, `data/`,
   `infra/`, `eval/`, `viz/`, `experiments/`, `reproducibility/` (each with a role docstring),
   `paths.py`, `py.typed`, version `2.0.0a0`. `pyproject.toml` rewritten: package
   `wildfire_marl`; core deps now `gymnasium` + `pettingzoo` (+ numpy/pandas/scipy/matplotlib/
   omegaconf/pyyaml/tqdm); SB3 dropped from core into an optional `baselines` extra (Phase-4
   Maskable-PPO); `marl` extra (torch + RLlib; final MARL lib choice is a Phase-5 decision);
   `geo` extra (rasterio et al. — GDAL via rasterio wheels / conda-forge); V1 CLI entry point
   removed.

4. **Ported the reusable core** (logic unchanged; import paths only — see `docs/MIGRATION.md`
   for the file-by-file mapping and the three deliberate adaptations):
   - `eval/`: `metrics.py` (ISR/WEL/CPS/RAC/PCA/PA/CCL/CE + core metrics, verbatim),
     `significance.py` (bootstrap CI, Cohen's d, Welch/paired/Wilcoxon — verbatim plus a SciPy
     ≥ 1.14 compatibility guard in `wilcoxon_test`: identical arrays now return NaN explicitly,
     preserving the V1 contract that newer SciPy silently broke),
     `transfer.py` (TRS/CDGG/adaptation-asymmetry verbatim; `transfer_matrix` now takes
     `evaluate_fn` as an argument instead of importing the retired V1 evaluation loop).
   - `reproducibility/`: `seeding.py`, `logging_utils.py`, `manifest.py`, and the four guard
     scripts converted to `python -m` modules — `check_seed_integrity`, `make_manifest`,
     `validate_tensors`, `build_report_tables` (V1 table/CSV registrations reset; V2 entries
     are registered per phase so V1 results can never be re-rendered).
   - `tests/`: `test_metrics.py`, `test_significance.py`, `test_seeding.py`, `test_logging.py`
     re-pointed at `wildfire_marl` (torch determinism test now `importorskip` — torch is an
     optional extra).

5. **Preserved the data.** `data/saudi_eastern_province/`, `data/california/`, `data/sample/`
   untouched (raw + processed + grid tensors — they feed Phase 2). `docs/data_card.md` kept,
   with a V2 provenance note added (V1 preprocessing references re-pointed into `legacy_v1/`;
   fuel model / CRS / ignition protocol to be re-documented in Phase 2).

6. **Documented the migration.** `docs/MIGRATION.md`: carries-over table (V1 path → V2 path →
   notes), retired table with reasons, the three AAAI rejection reasons fixed, platform note.
   `README.md` rewritten top-level: migration status, the single research bet, V2 layout,
   Linux-only quickstart, link to `legacy_v1/` as the honest-negative-result prototype.
   `environment-linux.yml` added (conda-forge; gcc/g++/make/cmake/Eigen + scientific core +
   rasterio/GDAL). `.github/workflows/ci.yml` re-pointed at `wildfire_marl` (lint/test/
   install-check/large-files/validate jobs; validate runs determinism + tensor validity + seed
   integrity + the ported significance/metric tests).

### Deviations from the plan (all documented in `docs/MIGRATION.md`)

- `transfer_matrix` decoupled from the V1 evaluation loop via an injected `evaluate_fn`
  (the V1 `evaluate.py` is env-coupled and is scheduled for porting in Phase 4; the pure
  TRS/CDGG/asymmetry functions are verbatim).
- `wilcoxon_test` gained an explicit all-zero-differences guard — SciPy 1.17 (Linux env) no
  longer raises `ValueError` there, which broke the ported test; the guard preserves the
  documented V1 behavior (NaN).
- `build_report_tables`/`check_seed_integrity` CSV registrations start empty in V2 (the V1
  entries referenced retired results; re-registered per phase from Phase 4 on).
- The whole `src/wildfire_rl` package was frozen (not only `envs/coordination/routing/
  ablation`): the reusable parts were *copied* out as ports, so leaving remnants behind would
  have created two live copies of the same code.

### Validation evidence (run on WSL2 Ubuntu 24.04, 2026-07-04)

```
$ python -c "import wildfire_marl; from wildfire_marl.eval import metrics, significance, transfer; print('core ported')"
core ported                                  (wildfire_marl 2.0.0a0, editable install)

$ pytest tests/
38 passed, 1 skipped (torch optional), 1 warning in 2.42s

$ python -m wildfire_marl.reproducibility.validate_tensors
OK  data/california/grids/32x32/state_tensor.npy  shape=(7, 32, 32)
OK  data/california/grids/64x64/state_tensor.npy  shape=(7, 64, 64)
OK  data/sample/state_tensor.npy  shape=(7, 8, 8)
OK  data/saudi_eastern_province/grids/32x32/state_tensor.npy  shape=(7, 32, 32)
OK  3 criticality rasters (california + saudi, [0,1])
TENSOR VALIDATION: OK

$ python -m wildfire_marl.reproducibility.check_seed_integrity
SEED INTEGRITY: OK        (no V2 checkpoints yet — nothing to certify, correctly skipped)

$ determinism: set_global_seed(0) twice -> identical draws   -> determinism OK
$ ruff check src tests && black --check src tests            -> all clean (23 files)

$ gcc --version   -> gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
$ cmake --version -> cmake version 3.28.3
$ make --version  -> GNU Make 4.3
$ g++ -I/usr/include/eigen3 check_eigen.cpp && ./a.out -> "eigen ok, det=-2"  (Eigen 3.4.0)
```

### Success Criteria (MANDATORY CHECKPOINT)

Technical verification
- [x] V1 fully relocated to `legacy_v1/` and frozen; new `wildfire_marl` package importable
- [x] Ported eval metrics + significance + reproducibility utilities pass their tests on Linux
      (38 passed / 1 optional-dep skip)
- [x] Build toolchain (gcc/make/cmake/Eigen) verified (trivial Eigen program compiled + ran)

Reproducibility
- [x] Region rasters + data card preserved and documented (tensor guard green on all of them;
      data card updated with V2 provenance note)

Scientific validity / framing
- [x] `docs/MIGRATION.md` states what carried over, what was retired, and the three rejection
      reasons it fixes

Platform
- [x] Development environment is Linux (WSL2 Ubuntu 24.04; native-Windows path abandoned for
      the sim) — `environment-linux.yml` committed for clean re-creation

### Proceed Rule

ALL boxes `[x]` → **proceed to Phase 1** (Cell2Fire + Firehose integration). Standing rule for
the whole V2 plan, now in force: **nothing from `legacy_v1/` is ever cited as a result** — it
is prior art / motivation only.
