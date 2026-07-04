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

---

## Phase 1 — Cell2Fire + Firehose Integration & Modernization

**Status:** ✅ COMPLETE · **Date:** 2026-07-04 · **Branch:** `v2-cell2fire` · **No commits made (user commits)**

### Objective (from the plan)

Build Cell2Fire, get a Firehose-style single-agent suppression episode stepping end-to-end on
stock maps, modernize the interface to Gymnasium, and profile per-step speed to size the project.

### Actions taken

1. **Simulator sources.**
   - `third_party/Cell2Fire/` — upstream Cell2Fire added as a **git submodule**
     (https://github.com/cell2fire/Cell2Fire, HEAD `b860bcc`); `.gitmodules` sets
     `ignore = untracked` so in-tree build artifacts don't dirty the submodule.
   - `third_party/firehose/` — Firehose **vendored** (https://github.com/aidan-curtis/firehose,
     commit `e49a52a`, 2022-05-12), pruned 193 MB → ~7 MB (dropped `.git`, `pretrained_models/`,
     `figs/`, `scratch/`, all maps except `Sub20x20`/`Sub40x40`/`Harvest40x40`). Provenance +
     pruning record in `third_party/firehose/VENDORED.md`; roles + rules in `third_party/README.md`.

2. **Physics-integrity audit (the "unmodified physics" evidence).** Diffed the Firehose C++
   fork against upstream at fork-base commit `336119d`:
   - byte-identical: `FBPfunc5_NoDebug.c`, `FBP5.0.h` (FBP fire-behavior equations),
     `SpottingFBP.cpp`, `Ellipse.cpp`, `Forest.cpp`, `Lightning.cpp`, `ReadCSV.cpp`, `WriteCSV.cpp`;
   - `CellsFBP.cpp`: 45 substantive changed lines, **all** `if (args->verbose)` →
     `if (should_print && args->verbose)` logging guards — zero fire-behavior changes;
   - `Cell2Fire.cpp` / `ReadArgs.*`: the interactive-control patch (`--steps-action`,
     `--steps-before`, `--HarvestPlan`, stdin protocol) — loop control + I/O only.
   Conclusion: the interactive binary's fire physics **is** unmodified Cell2Fire.

3. **Builds (WSL2 Ubuntu 24.04).** Installed `libboost-dev` (Cell2Fire includes
   `boost/algorithm/string.hpp`; Eigen alone is not enough — `environment-linux.yml` updated).
   Built **both** binaries with no source edits (`EIGENDIR` passed on the make command line):
   upstream `Cell2Fire` (stock) and the Firehose-fork `Cell2Fire` (interactive).

4. **Stock example.** Upstream binary on `Sub40x40`: exit 0, 1600-cell forest, 263 cells burnt
   (16.4 %), 8 per-period `ForestGrid*.csv` fire grids + final grid emitted.

5. **New wrapper code (all suppression/agent logic in the wrapper, simulator untouched):**
   - `src/wildfire_marl/env/cell2fire_binding.py` — `Cell2FireBinding`: spawns the interactive
     binary, waits for the `Input action` marker, writes 1-indexed harvest cells to stdin,
     collects `ForestGrid*.csv` paths, detects `Total Harvested Cells` (episode end). Improvements
     over Firehose's wrapper: seed is a parameter (Firehose hardcodes `--seed 123`), `--ROS-CV`
     defaults to 0.0 (deterministic spread; Firehose used 0.5), readline timeout guards, scratch
     I/O in `tempfile` dirs instead of the source tree.
   - `src/wildfire_marl/env/single_agent_env.py` — **Gymnasium** `FireSuppressionEnv`:
     obs `Box(0,1,(3,H,W))` (fire / harvested / fuel-mask channels), action `Discrete(num_cells)`
     with `action_diameter` 1|2 patches, `action_masks()` for Maskable-PPO (Phase 4), pluggable
     reward, seeded ignition sampling from `self.np_random`, `rgb_array` render, full
     reset/step/terminated/truncated semantics.
   - `src/wildfire_marl/env/rewards.py` — `Reward` ABC + `FireSizeReward` (ported Firehose
     objective: −cells-on-fire/total × 10) + declared `InfrastructureWeightedReward` hook that
     raises until Phase 3 lands (explicit error, not silent zero).
   - `scripts/profile_env.py` — the feasibility-gate profiler (see below).
   - `tests/test_env_smoke.py` — 3 integration tests (end-to-end step, harvest-registers-in-obs,
     seeded determinism); skip cleanly when the binary isn't built (CI-safe).

6. **`.gitignore` bug found & fixed.** The V1 virtualenv rules (`env/`, `ENV/`) matched *any*
   directory named `env` — silently ignoring the whole `src/wildfire_marl/env/` package
   (`env/__init__.py` was consequently **missing from the Phase-0 commit** `2f17a4c`). Rules are
   now root-anchored (`/env/`, `/ENV/`, `/venv/`, `/.venv/`); the four env-package files show as
   untracked and land in the next commit. An audit found no other silently-ignored live files.

### Deviations from the plan (documented)

- **Firehose's own Python was read, not executed.** Its `gym_env.py`/`evaluate_model.py` target
  dead `gym` 0.21 (incompatible with the Python 3.12 stack); the plan's own step 3 anticipates
  this rot and calls for a fresh Gymnasium binding. The stdin/stdout protocol was instead
  verified live through `Cell2FireBinding` (burn-in grids parsed; harvested cells confirmed in
  the state grid).
- **The interactive binary is the Firehose-fork build** — stock Cell2Fire cannot pause
  per-period for actions. Physics unchanged per the audit in item 2 (this is exactly the
  arrangement the plan's "modeled on Firehose `progress_to_next_state`" line implies).
- `--grid 32` in the profiler maps to the nearest stock map (`Sub40x40`); 32×32 instances are
  produced from region data in Phase 2.
- `libboost-dev` added to the toolchain (the plan listed only Eigen).

### Validation evidence (WSL2 Ubuntu 24.04, 2026-07-04)

```
$ ./Cell2Fire (stock, Sub40x40)         -> exit 0; 263/1600 cells burnt; 8 fire-grid CSVs
$ binding smoke                          -> burn-in 7 grids; state (40,40); no-op steps advance
                                            fire; harvest of 4 cells registers as -1 in grid
$ python -c "...FireSuppressionEnv...e.step(...)"   -> "env steps OK"   (plan's exact one-liner)
$ env validation                         -> reset obs (3,40,40) in space; 10 random steps,
                                            reward accumulates (-0.131); render() (40,40,3) uint8
$ determinism                            -> seed 7 twice + fixed 8-action script: identical
                                            ignition (1511), identical obs trajectories,
                                            identical rewards; seed 8 -> different ignition
$ pytest tests/                          -> 41 passed, 1 skipped (torch optional)
                                            (38 core + 3 new env integration tests)
$ ruff / black on src tests scripts      -> clean

$ python scripts/profile_env.py --grid 32 --episodes 5     (results/profile_env.csv + meta)
    mean step:  0.4 ms  (~2,375 steps/s)      mean reset (respawn): 0.033 s
    5 seeds x (1,000,000 train steps + 100 eval episodes) ~= 1.5 h single-process (~0.3 h/seed)
```

### Success Criteria (MANDATORY CHECKPOINT)

Technical verification
- [x] Cell2Fire builds and runs the stock example on Linux
- [x] Modern **Gymnasium** single-agent env steps end-to-end (reset/step/reward/render)
- [x] Cell2Fire physics **unmodified** — audited diff: FBP core byte-identical; fork changes are
      logging guards + IPC only; all suppression logic lives in the wrapper

Reproducibility
- [x] Deterministic given a seed (fixed ignition + fixed action sequence → identical trajectory;
      verified twice at seed 7, plus in `tests/test_env_smoke.py`)

Feasibility gate
- [x] Per-step / per-episode time measured and recorded (`results/profile_env.csv` +
      provenance meta): **0.4 ms/step, 33 ms/reset on Sub40x40** → a 5-seed study is ~1.5 h
      single-process. **Gate: PASS — no mitigation needed.** Caveat recorded: measured on 40×40
      with random actions and `/tmp` scratch I/O; re-profile after Phase-2 region landscapes
      (64×64) and before the Phase-5 MARL study.

### Proceed Rule

Speed gate passed decisively → **proceed to Phase 2** (geospatial data ingestion: Saudi +
California tensors → Cell2Fire landscapes with one shared fuel model).
