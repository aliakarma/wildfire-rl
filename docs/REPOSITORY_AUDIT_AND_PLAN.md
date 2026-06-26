# Wildfire-Saudi — Repository Engineering Audit & Open-Source Release Plan

> Deep technical audit and publication-grade GitHub repository design for the
> *Geospatial Wildfire Reinforcement Learning and Cross-Regional Generalization* project.
> Prepared as a NeurIPS-artifact-reviewer / reproducibility-auditor / senior-ML-systems review.
> All findings are grounded in the actual files present on disk (2026-06-26).

---

## 0. Scope of what was inspected

- `Complete Report.md` (524 lines, the narrative report)
- 19 notebooks (`Notebooks/`, `Notebooks/California/`, `Notebooks/Saudi/`), ~6,100 LOC of code cells
- `PyroRL_Saudi_Project/datasets/` (California + Saudi: raw, processed, grids @ 32×32 / 64×64)
- `PyroRL_Saudi_Project/models/` (13 PPO `.zip` checkpoints, **2.2 GB**)
- `PyroRL_Saudi_Project/results/` (8 CSVs)
- `PyroRL_Saudi_Project/figures/` (4 PNGs)
- Git state, remotes, and packaging/metadata files

---

# PHASE 1 — FULL PROJECT AUDIT

## 1.1 Critical, release-blocking findings (fix before *anything* else)

| # | Severity | Finding | Evidence |
|---|----------|---------|----------|
| C1 | 🔴 Blocker | **The git repository root is the user's home directory**, not the project. `git rev-parse --show-toplevel` → `C:/Users/Ali Akarma`. The entire home dir is one uncommitted repo. | `git status` lists `../../../.aws/`, `../../../.ssh`-class dirs, `.claude.json`, etc. |
| C2 | 🔴 Blocker | **`origin` points to an unrelated project**: `https://github.com/aliakarma/PPFL-Sensors.git`. Pushing would corrupt/clobber a different repo. | `git remote -v` |
| C3 | 🔴 Blocker | **No commits exist** ("No commits yet"). There is no version history, no provenance, no recoverable state. | `git log` empty |
| C4 | 🔴 Blocker | **2.2 GB of model binaries** committed-in-waiting. `models/` = 2.2 GB (11 × ~193 MB seed checkpoints + 2 legacy). GitHub hard-rejects >100 MB files; this repo cannot be pushed as-is. | `du -sh models` = 2.2G; each `*_100k_seed_*.zip` = 202 MB |
| C5 | 🟠 High | **Leaked Google Cloud / Earth Engine project ID** `gen-lang-client-0412024487` hardcoded in 4 notebooks. Not a secret key, but an identifying cloud resource that should not be public. | `ee.Initialize(project='gen-lang-client-0412024487')` in `03_dem_*`, `04_ndvi_*` (both regions) |
| C6 | 🟠 High | **No license, no README, no requirements, no `.gitignore`, no packaging.** A `find` for `readme/license/requirements/pyproject/environment.yml` returns nothing. The project is legally un-reusable and operationally un-runnable by anyone else. | filesystem scan |

> **Bottom line:** the project is currently at *negative* GitHub-readiness. The single most
> important action is to create a **fresh, scoped git repo inside the project folder** and never
> push the home-directory repo.

## 1.2 Per-notebook audit

Numbering is **inconsistent across regions** (see §1.4). Below, each notebook's role, I/O, and debt.

### Data pipelines (California `01–05`, Saudi `01–04`)

| Notebook | Purpose | Inputs | Outputs | Key debt |
|---|---|---|---|---|
| `California/01_era5_pipeline` / `Saudi/01_era5_pipeline` | Download ERA5 reanalysis → wind_x, wind_y, temperature, humidity grids @32/64 | CDS API (`cdsapi`), Drive | `wind_x.npy, wind_y.npy, temperature.npy, humidity.npy` | Hardcoded Drive paths (10×); `valid_time` vs `time` bug fixed inline (fragile, undocumented); no API-key handling shown |
| `02_firms_pipeline` (both) | FIRMS active-fire CSV → ignition mask `fire.npy` @32/64 | FIRMS CSV (`*_firms_june_2025.csv`) | `fire.npy` (32 & 64) | `GRID_SIZE` redefined per-cell (32 then 64) — copy-paste; FIRMS MAP_KEY flow not parameterized |
| `03_dem_pipeline` (both) | SRTM DEM → slope → `terrain.npy` | Earth Engine (`ee`) | `terrain.npy` + inline preview PNG (`dpi=300`) | **Leaked GEE project ID**; 18 Drive refs (Saudi); 2–3 MB of embedded base64 plot outputs bloating the `.ipynb` |
| `04_ndvi_pipeline` (both) | MODIS/NDVI → fuel density `fuel.npy` | Earth Engine | `fuel.npy` | Same GEE leak; embedded outputs; `saudi_ndvi.tif` (7.5 MB raw) committed |
| `California/05_california_tensor_stacking` | Stack 7 channels → `state_tensor.npy` (7,32,32)/(7,64,64) | all `*.npy` | `state_tensor.npy` | **No Saudi equivalent notebook** (asymmetry, see §1.4) |

### RL training / evaluation (Saudi `05–10`, California `06–07`, top-level `10`)

| Notebook | Purpose | Inputs | Outputs | Key debt |
|---|---|---|---|---|
| `Saudi/05_train_ppo_saudi_100k` | Train Saudi PPO, **5-seed loop**, 100k steps | `state_tensor.npy` | `ppo_saudi_32x32*.zip`, `multi_seed_results_saudi.csv` | Defines `SaudiWildfireEnv` + `CustomCNN` inline (732 LOC); env logic duplicated downstream |
| `Saudi/06_pyroRL_integration` | PyroRL sanity/integration check | pyrorl | none persisted | **97 LOC** — PyroRL is essentially *not* used by the actual experiments (see §1.5/novelty risk) |
| `Saudi/07a_experiments_and_ablations` | Ablations: no-wind / no-terrain / no-suppression / dense-fuel | env (redefined) | ablation CSV (var path) | `SaudiWildfireEnv` **redefined twice in one notebook**; writes a CSV whose name doesn't match the committed `phase6_ablation_results.csv` |
| `Saudi/07b_statistical_evaluation` | Multi-experiment stats + containment-curve figure; **only notebook with `set_seed()`** | env (redefined) | `multi_seed_statistics.csv`, `multi_seed_containment_curves.png` | Yet another `SaudiWildfireEnv`/`CustomCNN` copy |
| `Saudi/08_multi_agent_training` | Centralized cooperative PPO, `MultiAgentSaudiEnv` | env | model (var) | `total_timesteps=20000` — **5× shorter** than the 100k single-agent run; not comparable |
| `Saudi/09_marl_scaling_experiments_ipynb` | Scaling: 1/3/5 agents | env | scaling results (in-notebook) | Filename carries a stray `_ipynb`; 4th copy of `MultiAgentSaudiEnv`/`CustomCNN` |
| `Saudi/10_saudi_baseline_evaluation` | Baseline + cross-region eval table | `ppo_saudi*.zip` | `cross_region_evaluation.csv` | `SaudiEvaluationEnv` (5th env variant); produces CA→CA numbers that disagree with other files (§1.3) |
| `California/06_train_ppo_california` | Train CA PPO, 5-seed loop | `state_tensor.npy` | `ppo_california_32x32*.zip`, `multi_seed_results_california.csv` | **Hyperparameter drift inside the file**: one PPO block `batch_size=64`, another `batch_size=256`; 907 LOC |
| `California/07_zero_shot_transfer` | Saudi→CA zero-shot transfer | `ppo_saudi*.zip`, `ppo_california*.zip` | `zero_shot_transfer_results.csv` | `CaliforniaEvaluationEnv` (6th env variant); transfer is **one-directional only** |
| `10_final_analysis_and_figures` (top-level) | Aggregate CSVs → summary tables/figures | results CSVs | `final_summary_statistics.csv`, `publication_results_table.csv` | Two notebooks numbered `10`; pure aggregation, depends on every upstream CSV |

## 1.3 Experimental-integrity issues (publication risk)

1. **Three different "California native" reward numbers** for what should be one quantity:
   - `final_summary_statistics.csv`: **−86,180**
   - `cross_region_evaluation.csv` (CA→CA): **−85,326**
   - `zero_shot_transfer_results.csv` (CA→CA): **−82,614**
   There is no canonical evaluation; numbers come from different ad-hoc runs/seeds/episode counts. A reviewer will flag this immediately.
2. **Transfer matrix is incomplete & asymmetric.** Only `Saudi→Saudi`, `Saudi→California`, `California→California` exist. **`California→Saudi` is never evaluated**, yet the paper claims "cross-regional generalization." The claim is supported in one direction only.
3. **MARL metric is mislabeled.** Report §17 lists "1 Agent Mean 0.688, 3 Agent 0.629, 5 Agent 0.513" and says containment *improved ~25%*. The values **decrease**, so the metric must be *fraction of fire remaining* (lower = better). As written ("Mean"), it reads as the opposite of the claim. Needs explicit metric definition + direction.
4. **Non-comparable training budgets.** Single-agent = 100k steps; `08_multi_agent` = 20k steps. The report claims "identical training duration" — contradicted by the code.
5. **In-file hyperparameter drift** (`batch_size` 64 vs 256 in the CA trainer) undermines the "identical hyperparameters across regions" claim.
6. **Low statistical power**: N=5 seeds, 10 eval episodes, a single month (June 2025), one ROI per region. Saudi multi-seed burned-cell std (~30) is large relative to between-condition deltas.

## 1.4 Organization / naming issues

- **Cross-region numbering collision**: `Saudi/05`=training but `California/05`=tensor stacking; the two `10`s mean different things. Numbers don't encode pipeline stage consistently.
- **Missing Saudi tensor-stacking notebook** (California has `05_california_tensor_stacking`; Saudi's stacking is implicit/absent).
- **Filename hygiene**: `09_marl_scaling_experiments_ipynb.ipynb` (stray suffix); `Complete Report.md` (space in name).
- **Dataset asymmetry** despite "identical structure" claim:
  - Saudi `grids/64x64/` **lacks `state_tensor.npy`** (CA has it for both resolutions).
  - Metadata JSONs (`state_metadata`, `fuel_metadata`, `terrain_metadata`) exist for **Saudi only**; California has **none**.
  - Saudi has extra `processed/risk/` and `raw/osm/petro_sites/` (the latter **empty**); California has neither.
- **6 near-duplicate env classes** (`SaudiWildfireEnv`, `CaliforniaWildfireEnv`, `SaudiEvaluationEnv`, `CaliforniaEvaluationEnv`, `MultiAgentSaudiEnv`, plus ablation variant) and **~7 copies of `CustomCNN`**. There is no single source of truth for the environment — the "identical architecture" guarantee is unverifiable.

## 1.5 Naming / novelty risk

The project is named **PyroRL_Saudi_Project**, but the experiments do **not** run on PyroRL — every environment is a hand-rolled `gym.Env`. PyroRL appears only in a 97-LOC sanity notebook. Positioning the work as "PyroRL-based" is misleading and a reviewer/maintainer will notice. Either (a) genuinely build on PyroRL, or (b) rename and describe it as "inspired by PyroRL."

## 1.6 Obsolete / orphaned artifacts

Not generated by any current notebook (likely from deleted earlier versions):
- `results/phase6_ablation_results.csv` (07a writes a differently-named file)
- `figures/agent_trajectory.png`, `figures/fuel_density_comparison.png`, `figures/wildfire_temporal_propagation.png`
- empty dir `datasets/saudi_eastern_province/raw/osm/petro_sites/`

Also note `agent_trajectory.png` and `wildfire_temporal_propagation.png` are **byte-identical size (15,128 B)** — possibly a duplicated/placeholder export.

## 1.7 Documentation & metadata gaps

- Zero markdown cells in all 11 data-pipeline notebooks (pure code, no narrative).
- No data dictionary, no units, no CRS/projection statement, no temporal coverage manifest.
- No hyperparameter table, no reward-function spec, no env-dynamics spec in machine-readable form.
- `Complete Report.md` has **no figures, no equations, no related-work, no hyperparameter table** — it is a prose summary, not a paper draft.

---

## 1.8 The four assessments

### Repository health — **2/10**
Wrong git root, wrong remote, no commits, 2.2 GB of binaries, no scaffolding, heavy duplication, embedded notebook outputs. Structurally unsound for collaboration.

### Reproducibility — **3/10**
A reader *could* reconstruct results conceptually (clear channel ordering, documented ROIs, multi-seed intent). But: unpinned deps, Colab-only paths, one seed-helper used in one notebook, divergent result numbers, non-comparable budgets, no environment lock, no checksums, no `make reproduce`. A NeurIPS artifact badge is not currently attainable.

### Publication-readiness — **4/10**
Genuinely interesting research questions and a real cross-regional result. But integrity gaps (inconsistent numbers, one-directional transfer, metric mislabeling, budget mismatch), small N, single-month data, and a misleading "PyroRL" framing would draw major-revision reviews.

### GitHub-readiness — **1/10**
Cannot be pushed (size + wrong remote), no license, no docs, no CI. Nothing a visitor could clone and run.

---

# PHASE 2 — RESEARCH PIPELINE MAPPING

## 2.1 Actual execution order & artifact flow

```
PER REGION (run for Saudi and California independently)
  01_era5_pipeline      ──> wind_x.npy, wind_y.npy, temperature.npy, humidity.npy
  02_firms_pipeline     ──> fire.npy            (ignition mask, 32 & 64)
  03_dem_pipeline       ──> terrain.npy         (slope from SRTM)   [GEE]
  04_ndvi_pipeline      ──> fuel.npy            (NDVI→fuel)         [GEE]
        │
        ▼
  05_tensor_stacking    ──> state_tensor.npy (7,G,G)
   (California explicit; Saudi MISSING/implicit)

SAUDI RL TRACK
  05_train_ppo_saudi_100k ──> ppo_saudi_32x32*.zip, multi_seed_results_saudi.csv
  06_pyroRL_integration   ──> (sanity only)
  07a_experiments_ablations ──> ablation CSV (orphaned naming)
  07b_statistical_eval     ──> multi_seed_statistics.csv, multi_seed_containment_curves.png
  08_multi_agent_training  ──> MARL model (20k steps)
  09_marl_scaling          ──> 1/3/5-agent scaling results
  10_saudi_baseline_eval   ──> cross_region_evaluation.csv   (needs ppo_saudi + ppo_california)

CALIFORNIA RL TRACK
  06_train_ppo_california  ──> ppo_california_32x32*.zip, multi_seed_results_california.csv
  07_zero_shot_transfer    ──> zero_shot_transfer_results.csv (needs BOTH region models)

AGGREGATION
  10_final_analysis_and_figures ──> final_summary_statistics.csv, publication_results_table.csv
        (consumes all results CSVs)
```

## 2.2 Dependency graph (stage view)

```
                 ┌─────────── raw downloads (CDS / FIRMS / GEE) ───────────┐
                 ▼                  ▼                ▼               ▼
              era5(01)          firms(02)        dem(03)         ndvi(04)
                 └────────┬─────────┴────────┬───────┴──────────────┘
                          ▼                   ▼
                    state_tensor (05 stacking, per region)
                          │
              ┌───────────┼────────────────────────────┐
              ▼           ▼                              ▼
        saudi PPO(05)  california PPO(06)        ablations/stats(07a/07b)
              │           │                              │
              ▼           ▼                              ▼
       MARL(08/09)   zero-shot transfer(07)      containment figure
              │           │                              │
              └─────► baseline/cross-region(10) ◄────────┘
                          │
                          ▼
              final analysis & figures (top-level 10)
```

**Cross-track coupling**: `Saudi/10` and `California/07` both require **both** region models → the two tracks are *not* independent and must be trained before either evaluation runs. This is implicit and undocumented today.

## 2.3 Architectural problems (and the fix)

| Problem | Consequence | Fix (Phase 3 design) |
|---|---|---|
| Env + CNN copied 7× | "Identical architecture" unverifiable; silent drift | Single `wildfire_rl/envs/` + `wildfire_rl/models/cnn.py` imported everywhere |
| Region encoded in class names & paths | Can't add a 3rd region without new notebooks | One parameterized `WildfireEnv(region_cfg)` + YAML region configs |
| Colab Drive paths everywhere | Runs only on the author's Drive | Central `paths.py` + `configs/paths.yaml`, env-var overridable |
| Numbers regenerated ad-hoc | Inconsistent published values | One `evaluate.py` writing one canonical results table with run metadata |
| Notebooks are the pipeline | Not automatable, not testable | Logic → `src/`; notebooks become thin demos calling `src` |

---

# PHASE 3 — GITHUB REPOSITORY DESIGN

## 3.1 Target structure

```
wildfire-rl/                         # repo root == project root (NOT home dir)
├── README.md
├── LICENSE                          # MIT or Apache-2.0
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── pyproject.toml                   # single source of build + deps + tooling
├── requirements.txt                 # pinned, generated from lock
├── requirements-dev.txt
├── environment.yml                  # conda (geospatial stack: gdal/rasterio)
├── Makefile                         # make data | train | evaluate | figures | reproduce
├── Dockerfile / .dockerignore
├── .gitignore
├── .gitattributes                   # Git LFS rules
├── .pre-commit-config.yaml
├── .github/
│   ├── workflows/{ci.yml, lint.yml, release.yml}
│   ├── ISSUE_TEMPLATE/ , PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
│
├── src/wildfire_rl/                 # THE installable package (pip install -e .)
│   ├── __init__.py        (+ __version__)
│   ├── config.py                    # pydantic/OmegaConf schema loader
│   ├── paths.py                     # all path resolution, env-overridable
│   ├── data/
│   │   ├── era5.py  firms.py  dem.py  ndvi.py     # one module per source
│   │   ├── tensor_stack.py          # the ONE stacking implementation
│   │   └── manifest.py              # checksums + provenance
│   ├── envs/
│   │   ├── base.py                  # WildfireEnv(gym.Env) — single source of truth
│   │   ├── single_agent.py
│   │   └── multi_agent.py           # MultiAgentWildfireEnv
│   ├── models/cnn.py                # the ONE CustomCNN
│   ├── train/{ppo.py, callbacks.py, seeding.py}
│   ├── eval/{evaluate.py, transfer.py, baselines.py, metrics.py}
│   ├── viz/figures.py
│   └── cli.py                       # `wildfire-rl <subcommand>`
│
├── configs/                         # Hydra/OmegaConf
│   ├── config.yaml
│   ├── region/{saudi.yaml, california.yaml}
│   ├── env/{single.yaml, multi.yaml}
│   ├── train/ppo.yaml
│   └── experiment/{transfer.yaml, ablation.yaml, scaling.yaml, multiseed.yaml}
│
├── scripts/                         # thin CLI entrypoints / repro orchestration
│   ├── download_data.py  build_tensors.py  train.py
│   ├── evaluate.py  run_transfer.py  make_figures.py
│   └── fetch_models.py              # pull checkpoints from Zenodo/HF
│
├── notebooks/                       # demos only, output-stripped, papermill-able
│   ├── 00_quickstart_demo.ipynb
│   ├── 01_data_pipeline_walkthrough.ipynb
│   ├── 02_train_and_evaluate.ipynb
│   └── 03_cross_region_transfer.ipynb
│
├── tests/                           # pytest
│   ├── test_env_api.py  test_tensor_stack.py
│   ├── test_seeding.py  test_metrics.py
│   └── test_cli_smoke.py
│
├── data/                            # NOT in git (LFS pointers or download)
│   ├── raw/{saudi,california}/...
│   ├── processed/...
│   └── grids/.../state_tensor.npy
│
├── models/                          # NOT in git — Zenodo/HF release
├── results/                         # canonical CSVs (small → may commit)
├── figures/                         # generated → .gitignore (keep paper-final in docs/)
└── docs/
    ├── index.md  data_card.md  model_card.md
    ├── reproducibility.md  environment_spec.md
    ├── paper/ (report + final figures)
    └── architecture.md
```

## 3.2 Conventions & strategies

- **Naming**: `snake_case` modules; notebooks `NN_topic.ipynb` where `NN` = global pipeline stage (no per-region renumbering); artifacts `{region}_{quantity}_{grid}.{ext}`.
- **Versioning**: SemVer for code (`pyproject` `__version__`), CalVer-style data releases (`data-2025.06`), and a Zenodo DOI per tagged release. Tag `v0.1.0` = MVP.
- **Datasets**: raw never in git (download scripts + manifest of checksums); derived tensors via Git LFS *or* regenerable by `make data`. Ship a tiny `data/sample/` for CI/demo.
- **Checkpoints**: never in git; published to Hugging Face Hub + Zenodo; `fetch_models.py` resolves by version + sha256.
- **Experiment tracking**: Weights & Biases or MLflow (offline mode acceptable); every run writes a `run_metadata.json` (git sha, config hash, seed, lib versions, timestamp).
- **Seeds**: one `seeding.set_global_seed(seed)` (python/numpy/torch + `PYTHONHASHSEED` + SB3 `set_random_seed`), called in every entrypoint; seeds declared in config, not literals.
- **Config**: Hydra/OmegaConf; no magic numbers in code. Region/env/train/experiment composition.
- **Logging**: stdlib `logging` (no bare prints), structured run logs under `results/logs/`.
- **Figures**: deterministic `make figures` from canonical CSVs; matplotlib style file committed; PNG+PDF, `dpi=300`.

---

# PHASE 4 — FILE-BY-FILE IMPLEMENTATION PLAN

## 4.1 New files to create (what / why / where)

| File | Why | Contents (essentials) |
|---|---|---|
| `README.md` | First impression; runnable in <5 min | Title+subtitle, badges, 1 hero figure, abstract, install, `make reproduce`, results table, citation, license |
| `LICENSE` | Legal reuse | **Code**: MIT or Apache-2.0. Note data licenses (ERA5/Copernicus, MODIS/NASA, FIRMS, SRTM) separately in `docs/data_card.md` |
| `pyproject.toml` | Single build/dep/tooling source | `[project]` deps **pinned** (gymnasium, stable-baselines3, torch, rasterio, rioxarray, xarray, cdsapi, earthengine-api, geemap, numpy, pandas, matplotlib); `[tool.ruff/black/pytest/mypy]`; entrypoint `wildfire-rl = wildfire_rl.cli:main` |
| `requirements.txt` / `-dev.txt` | Non-packaging installs / CI | Pinned, hash-locked (pip-tools) |
| `environment.yml` | Geospatial stack is painful via pip | conda-forge gdal/rasterio/proj + pip extras |
| `.gitignore` | Stop 2 GB leak | `models/*.zip`, `data/raw/**`, `data/processed/**`, `*.nc`, `*.tif`, `*.npy` (except `data/sample/**`), `figures/*.png` (except docs), `.ipynb_checkpoints/`, `__pycache__/`, `.env`, creds |
| `.gitattributes` | LFS | `*.npy *.tif *.nc filter=lfs` (if LFS chosen for derived tensors) |
| `CITATION.cff` | Citable; GitHub renders "Cite this repository" | authors, title, version, DOI placeholder |
| `CONTRIBUTING.md` | Onboard collaborators | branch model, pre-commit, test req, env setup |
| `CODE_OF_CONDUCT.md` | Community norm | Contributor Covenant 2.1 |
| `Makefile` | One-command reproducibility | `data`, `tensors`, `train`, `evaluate`, `transfer`, `figures`, `test`, `reproduce` |
| `Dockerfile` | Pinned runtime for artifact eval | CUDA-optional base + conda env + `pip install -e .` |
| `.github/workflows/ci.yml` | Trust signal + regression guard | lint + `pytest` on `data/sample/`, smoke-train 1k steps, env-API conformance |
| `docs/data_card.md`, `model_card.md` | NeurIPS datasheet/model-card norm | sources, licenses, CRS, temporal coverage, intended use, limitations, biases |
| `docs/reproducibility.md` | Artifact reviewers read this first | exact steps, expected numbers ± tolerance, hardware, runtime, seeds |

## 4.2 Refactor / move / merge / delete

**Become Python modules (logic extracted from notebooks):**
- All 6 env classes → `src/wildfire_rl/envs/{base,single_agent,multi_agent}.py` (one canonical env, region via config).
- All `CustomCNN` copies → `src/wildfire_rl/models/cnn.py`.
- `normalize()` (duplicated in 01/03/04) → `src/wildfire_rl/data/_utils.py`.
- ERA5/FIRMS/DEM/NDVI cells → `src/wildfire_rl/data/{era5,firms,dem,ndvi}.py`.
- Tensor stacking → `src/wildfire_rl/data/tensor_stack.py` (fixes the missing-Saudi-stacking gap).
- Training/eval loops → `src/wildfire_rl/train/ppo.py`, `eval/{evaluate,transfer,baselines}.py`.

**Merge:**
- `Saudi/07a` + `07b` → one `eval/ablation.py` + `scripts/run_ablation.py`.
- `Saudi/08` + `09` → one `train/multi_agent` driven by `configs/experiment/scaling.yaml`.
- `Saudi/10` + `California/07` → one `eval/transfer.py` producing the **full 2×2 transfer matrix** (fixes the missing CA→Saudi cell).

**Rename:**
- `09_marl_scaling_experiments_ipynb.ipynb` → drop stray `_ipynb`.
- `Complete Report.md` → `docs/paper/report.md`.
- Region notebooks → global-stage numbering.

**Delete (after confirming with author):**
- Orphaned `results/phase6_ablation_results.csv`, `figures/{agent_trajectory,fuel_density_comparison,wildfire_temporal_propagation}.png` (regenerate canonically), empty `raw/osm/petro_sites/`.
- `Saudi/06_pyroRL_integration` → demote to `notebooks/`/`docs/` appendix or drop (it isn't on the experiment path).

**Keep as demo notebooks (output-stripped):** thin walkthroughs that import `src/`.

---

# PHASE 5 — NOTEBOOK REFACTORING STRATEGY

| Current notebook | Fate |
|---|---|
| `01–04` data pipelines (both regions) | **→ modules** (`src/wildfire_rl/data/*`); one demo notebook `01_data_pipeline_walkthrough` |
| `05_california_tensor_stacking` (+ missing Saudi) | **→ module** `tensor_stack.py`, region-parameterized |
| `Saudi/05`, `California/06` training | **→ module + `scripts/train.py`**; demo `02_train_and_evaluate` |
| `06_pyroRL_integration` | **Exploratory only** → appendix/remove |
| `07a/07b` ablation+stats | **→ reproducible pipeline** (`scripts/run_ablation.py`) |
| `08/09` MARL | **→ module + scaling config** |
| `Saudi/10`, `California/07` transfer/baseline | **→ `eval/transfer.py`**; demo `03_cross_region_transfer` |
| top-level `10_final_analysis` | **→ `viz/figures.py` + `make figures`**; keep a publication notebook |

**Standards**: notebooks contain *no* business logic, only `from wildfire_rl import ...`; outputs stripped on commit (`nbstripout` pre-commit); parameterized via **papermill** + tagged `parameters` cell; `jupytext` pairing (`*.py:percent`) so diffs are reviewable; Hydra configs drive everything heavyweight. Naming: `NN_topic.ipynb`, global stage order.

---

# PHASE 6 — DATASET & MODEL MANAGEMENT

| Asset | In git? | Where it lives | How obtained |
|---|---|---|---|
| Raw ERA5 `.nc`, FIRMS `.csv`, DEM/NDVI `.tif` | ❌ No | Zenodo archive + `scripts/download_data.py` | Re-download from CDS/FIRMS/GEE (keys via env) |
| Derived `.npy` grids / `state_tensor` | ⚠️ Optional (Git LFS) or regenerate | LFS *or* `make tensors` | Deterministic from raw |
| `data/sample/` (one tiny tensor) | ✅ Yes (<1 MB) | git | For CI + quickstart |
| PPO checkpoints (2.2 GB) | ❌ **Never** | **Hugging Face Hub** + **Zenodo** | `scripts/fetch_models.py` by version+sha256 |
| Canonical results CSVs (small) | ✅ Yes | `results/` | `make evaluate` |
| Figures | ❌ generated (keep paper-final in `docs/paper/`) | `make figures` | deterministic |

**Strategy details**
- **Git LFS** only if you want one-clone reproducibility for derived tensors; otherwise prefer *regeneration* + a Zenodo data bundle (cleaner history, no LFS quota pain).
- **Hugging Face**: model repo `aliakarma/wildfire-rl-ppo` with a **model card** (training data, seeds, metrics, intended use, limitations). Tag checkpoints by `region/grid/steps/seed`.
- **Zenodo**: link the GitHub repo → cut a release → mint a **DOI**; archive (a) code snapshot, (b) raw+derived data bundle, (c) checkpoints. Put DOI in `README` badge + `CITATION.cff`.
- **Reproducibility snapshots**: each release ships `environment-lock.txt`, `run_metadata.json` per result, and a `MANIFEST.sha256` for all data/models.
- **Auto-generated**: tensors, results, figures (all via `make`). **Downloadable**: raw data, checkpoints. **Tracked**: code, configs, docs, tiny sample, canonical small CSVs.

---

# PHASE 7 — OPEN-SOURCE & RESEARCH POSITIONING

**Public strength today**: weak (un-runnable). **After Phase 3–4**: strong and differentiated — geospatially-grounded wildfire RL with a *real cross-regional (desert vs. forest) transfer study* is genuinely novel and under-explored.

**Positioning / novelty pitch**
- Lead with the **cross-regional ecological transfer** result (Saudi desert ↔ California forest) — that's the differentiator, not "another fire RL env."
- Frame as a **benchmark + framework**: reproducible multi-seed environment, standardized 7-channel geospatial tensors, transfer protocol.
- Be honest about PyroRL ("inspired by") and centralized-vs-decentralized MARL.

**Showcase**
- Hero animation: `wildfire_temporal_propagation` as a GIF (fire spread + agent suppression overlay).
- `multi_seed_containment_curves.png` (already strong) as the headline result figure.
- A **2×2 transfer matrix heatmap** (once CA→Saudi is added).
- `fuel_density_comparison.png` to motivate the ecological contrast.
- A benchmark table (Saudi vs California native + transfer) front-and-center in README.

**Repo metadata**
- **Title**: `wildfire-rl` (or `pyrofire-transfer`)
- **Subtitle**: *"Geospatial reinforcement learning for wildfire suppression and cross-regional policy transfer (Saudi Arabia ↔ California)."*
- **Topics/tags**: `reinforcement-learning`, `wildfire`, `ppo`, `multi-agent-rl`, `transfer-learning`, `gymnasium`, `stable-baselines3`, `geospatial`, `remote-sensing`, `era5`, `firms`, `ndvi`, `domain-generalization`, `benchmark`.
- **Release naming**: `v0.1.0-mvp`, `v0.2.0-marl`, `v1.0.0-paper`; data `data-2025.06`; DOI per tag.
- **Collaborator magnets**: "good first issue" labels (add a 3rd region, decentralized MARL), a clear `docs/architecture.md`, leaderboard-style results table, and a Colab "Open in Colab" badge on the demo notebook.

---

# PHASE 8 — FINAL DELIVERABLES

## 8.1 Proposed repository tree
See **§3.1** (canonical target structure).

## 8.2 Prioritized implementation roadmap

**P0 — De-risk (hours):**
1. `git init` **inside** the project folder; set correct remote (new `wildfire-rl` repo). Never push the home repo.
2. Add `.gitignore` excluding `models/`, `data/raw|processed`, `*.zip/*.tif/*.nc/*.npy`.
3. Scrub the GEE project ID (`gen-lang-client-0412024487`) → read from env var; strip from notebook history.
4. `nbstripout` all notebooks (kills embedded-output bloat, MBs → KBs).

**P1 — Make it real (days):**
5. Add `LICENSE`, `README`, `pyproject.toml`, `requirements*.txt`, `environment.yml`.
6. Extract the **one** env + **one** CNN into `src/`; delete the 6 copies.
7. Central `paths.py`/config; remove Drive paths.
8. Move checkpoints to HF/Zenodo; add `fetch_models.py`.

**P2 — Reproducible (1–2 weeks):**
9. Hydra configs + `scripts/` + `Makefile reproduce`.
10. Canonical `evaluate.py` → single results table (kills the −86k/−85k/−82k discrepancy).
11. Complete the **2×2 transfer matrix**; equalize training budgets; fix metric labels.
12. `tests/` + CI on `data/sample/`.

**P3 — Polished/paper (weeks):**
13. `docs/` (data card, model card, reproducibility, architecture).
14. Zenodo DOI, model cards, demo notebooks + Colab badges, hero GIF.
15. W&B/MLflow tracking; expand seeds/episodes/months for statistical power.

## 8.3 Minimum Viable Public Release (`v0.1.0`)
- Correct repo + remote, `.gitignore`, MIT license, README with one hero figure + results table.
- `src/wildfire_rl` with the single env/CNN; `pip install -e .` works.
- `data/sample/` + quickstart notebook runs end-to-end in Colab.
- Checkpoints on Hugging Face; `fetch_models.py`.
- Canonical `results/` CSVs (one consistent set).
- Basic CI (lint + smoke test).

## 8.4 Research-grade polished release (`v1.0.0-paper`)
- Full Hydra config system, `make reproduce` regenerates every number/figure within tolerance.
- Complete 2×2 transfer matrix, equalized budgets, ≥10 seeds, multi-month data, power analysis.
- Zenodo DOI, data card + model card, decentralized-MARL stub, docs site, hero animation.
- Paper draft in `docs/paper/` with embedded figures, hyperparameter tables, related work.

## 8.5 Risk assessment before open-sourcing

| Risk | Severity | Mitigation |
|---|---|---|
| Pushing the **home-dir repo** (leaks `.aws`, `.ssh`, `.claude.json`, browser data) | 🔴 Catastrophic | New scoped repo; verify `git rev-parse --show-toplevel` is the project |
| Pushing to **PPFL-Sensors** remote (clobbers another project) | 🔴 High | Reset `origin` to the new repo before first push |
| 2.2 GB binaries → rejected push / bloated history | 🔴 High | `.gitignore` + LFS/Zenodo; never `git add models/` |
| Leaked GEE project ID | 🟠 Medium | Env var + history scrub |
| Data-license compliance (ERA5/MODIS/SRTM/FIRMS redistribution terms) | 🟠 Medium | Don't redistribute raw; link sources; document in data card |
| Result inconsistencies discovered post-release | 🟠 Medium | Canonical eval before tagging `v1.0.0` |
| Overstated "PyroRL"/"MARL" claims | 🟡 Low-Med | Reframe as "inspired by" / "centralized cooperative PPO" |

## 8.6 Publication / reviewer-readiness checklist

- [ ] Single canonical results table; all reported numbers reproduce from one `evaluate` run
- [ ] Complete & symmetric transfer matrix (Saudi↔California, both directions)
- [ ] Equal training budgets across compared conditions; hyperparameters in one table
- [ ] Metric definitions explicit (containment vs. fire-remaining; direction of "better")
- [ ] Seeds fixed & reported; ≥10 seeds, ≥30 eval episodes; CIs not just std
- [ ] One environment implementation; documented dynamics + reward function
- [ ] Data card (sources, CRS, temporal coverage, licenses, limitations)
- [ ] Model card (training data, compute, metrics, intended use, biases)
- [ ] `make reproduce` from clean clone → matches reported numbers within tolerance
- [ ] Pinned environment + Dockerfile; hardware + runtime documented
- [ ] DOI (Zenodo); CITATION.cff; license; CI green
- [ ] Honest framing of PyroRL relationship and centralized-MARL scope

---

*End of audit. Generated from a file-level inspection of the repository on 2026-06-26.*
