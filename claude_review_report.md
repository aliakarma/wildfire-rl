# 🧠 OVERALL VERDICT

**Score: 5.5/10**
**Decision: Major Revision**

The single most serious flaw is that **all 8 result CSV files under `results/legacy_notebook_runs/` are pre-baked artifacts from prior notebook runs with no generation script in the current codebase**—they cannot be traced to any live computation and form the basis of every quantitative claim in the report, yet the canonical pipeline that *could* produce verifiable numbers has never been run end-to-end (no `results/runs/` metadata exists, and no canonical result CSVs are present). The codebase architecture is genuinely well-refactored (single-source-of-truth envs, metrics, configs), the seeding infrastructure is above average for an RL project, and the test suite covers the critical determinism contract. However, **no baseline comparison exists in any result file** (Random/NoOp policies were added in the refactor but never evaluated against legacy results), the transfer matrix is **incomplete** (California→Saudi row missing from legacy CSVs), normalization is applied per-region making transfer reward comparisons formally invalid, and **zero statistical significance tests** are conducted. The work is a promising research framework, not yet a publishable result.

---

# 🚨 CRITICAL ISSUES (rejection-level)

## Issue 1: All quantitative results are untraceable legacy artifacts

- **Problem**: Every reported number originates from `results/legacy_notebook_runs/*.csv`—files that were produced by now-superseded notebooks, not by the canonical pipeline (`wildfire-rl train/evaluate/transfer`). No `results/runs/*.json` metadata files exist, so no result can be tied to a specific code version, config hash, or seed.
- **Evidence**: [results/README.md](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/results/README.md#L8-L10) explicitly states these are *"the original notebook CSVs, kept for provenance. Superseded by the canonical pipeline"*. The `results/runs/` directory does not exist.
- **Why it invalidates results**: A reviewer cannot verify that the claimed metrics (e.g. Saudi reward −40525 ± 773) were produced by the code in this repository. The code has been materially refactored (vectorized seeded spread instead of per-neighbor draws, pooling added to the CNN changing the model), so the legacy numbers are unlikely to be bit-reproducible even with the same data and seeds.
- **Exact fix**:
  ```bash
  # Run the canonical pipeline end-to-end and commit the outputs:
  make reproduce   # test → train → evaluate → transfer → figures
  # This will write results/runs/*.json, results/eval_*.json, results/transfer_matrix.csv
  # Commit these canonical artifacts and deprecate or remove legacy_notebook_runs/
  ```

---

## Issue 2: No baseline comparison in any result table

- **Problem**: The `RandomPolicy` and `NoOpPolicy` baselines were added in the refactor ([baselines.py](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/eval/baselines.py)) but **no result file contains baseline numbers**. All CSVs contain only PPO agent metrics. Without baselines, there is zero evidence the learned policy outperforms doing nothing.
- **Evidence**: [baselines.py](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/eval/baselines.py) docstring line 3–4: *"The original 'baseline_evaluation' notebook contained NO baseline — only the PPO model evaluated on its own region."* The 8 CSVs in `legacy_notebook_runs/` confirm this—none contain a "random" or "noop" row.
- **Why it invalidates results**: The claim that PPO learns effective suppression (report §20-21) is unfalsifiable without controls. The reward of −40525 for Saudi might be *worse* than a random agent—we simply don't know.
- **Exact fix**:
  ```bash
  # Run evaluation with baselines for each region:
  wildfire-rl evaluate --config configs/experiment/multiseed.yaml
  wildfire-rl evaluate --config configs/experiment/multiseed.yaml \
      --set region.name=california region.dir=california
  # Include baseline rows in the publication results table
  ```

---

## Issue 3: Incomplete transfer matrix — California→Saudi missing

- **Problem**: The [cross_region_evaluation.csv](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/results/legacy_notebook_runs/cross_region_evaluation.csv) contains only 3 of 4 required cells: Saudi→Saudi, Saudi→California, California→California. The **California→Saudi** cell is absent. Without this, the transfer claim is asymmetric and scientifically incomplete.
- **Evidence**: The CSV has exactly 3 data rows (line 2–4). The refactored [transfer.py](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/eval/transfer.py) line 39–41 computes the full N×N matrix, but has never been run (no `results/transfer_matrix.csv` exists).
- **Why it invalidates results**: The report (§22) claims *"Policies trained under Saudi ecological conditions do not generalize effectively to California"* — but the reverse direction was never tested, making the claim about ecological specialization one-directional and incomplete.
- **Exact fix**:
  ```bash
  wildfire-rl transfer --config configs/experiment/transfer.yaml
  # Produces the full 4-cell matrix in results/transfer_matrix.csv
  ```

---

## Issue 4: Per-region normalization invalidates cross-region reward comparison

- **Problem**: State tensors are min-max normalized **per-region independently** ([normalize.py](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/data/normalize.py#L4-L6), [data_card.md](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/docs/data_card.md#L42-L44)). The default `reward_mode="raw"` computes reward as `−Σ fire` ([base.py](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/envs/base.py#L114)). Since the fire channel's absolute scale differs between regions, comparing Saudi reward (−40525) to California reward (−86180) and attributing the difference to ecological difficulty is **confounded by normalization scale**.
- **Evidence**: [config.py](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/config.py#L62-L64) documents this: `"raw": -sum(fire) (preserves original behavior, NOT comparable across regions)`. Yet the report §20–21 directly compares these numbers, and the transfer config ([transfer.yaml](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/configs/experiment/transfer.yaml#L17)) also uses `reward_mode: raw`.
- **Why it invalidates results**: The claim that California is "substantially more difficult" (§21) could be an artifact of California having more initially burning cells (145 vs 110 nonzero in fire channel), not of genuine ecological difficulty. The framework has `reward_mode="normalized"` and `fit_shared_minmax()` specifically to address this, but neither is used in any reported result.
- **Exact fix**: Re-run transfer and multi-seed experiments with `reward_mode: normalized` and shared normalization. Report both raw and normalized numbers.

---

## Issue 5: Zero statistical rigor — no significance tests, no confidence intervals, no effect sizes

- **Problem**: All results are reported as mean ± std over 5 seeds with no statistical significance tests, no confidence intervals, no effect sizes, and no ablation significance.
- **Evidence**: Every CSV uses exactly this format: e.g., [publication_results_table.csv](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/results/legacy_notebook_runs/publication_results_table.csv) — `"-40525.74 ± 772.68"`. No p-values, no t-tests, no Cohen's d anywhere in the codebase or results.
- **Why it invalidates results**: With N=5 seeds and large standard deviations (reward std ~773 to ~1278), no comparison (PPO vs baseline, Saudi vs California, transfer vs native) is statistically grounded. The claimed "100% relative reward difference" in transfer (§22) could be noise.
- **Exact fix**: Add scipy t-tests (or Wilcoxon) comparing PPO vs Random, PPO vs NoOp, native vs transferred, and each ablation variant vs baseline. Report p-values and 95% CIs for all primary metrics.

---

# ⚠️ MODERATE ISSUES

## Issue 6: Saudi tensor dtype inconsistency (float64 vs float32)
- **Problem**: The Saudi state tensor is stored as `float64` (57,472 bytes for 7×32×32) while California is `float32` (28,800 bytes). The environment casts to float32 at [base.py L44](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/envs/base.py#L44), so this is silently truncated. The tensor_stack module produces float32 ([tensor_stack.py L46](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/data/tensor_stack.py#L46)), meaning the Saudi tensor was built by an older process.
- **Fix**: Rebuild the Saudi state tensor with the canonical pipeline, or cast and re-save: `np.save(path, np.load(path).astype(np.float32))`.

## Issue 7: California state_metadata.json missing
- **Problem**: `data/saudi_eastern_province/grids/32x32/state_metadata.json` exists, but `data/california/grids/32x32/state_metadata.json` does not. This suggests the California tensor was also built by a different (older) process.
- **Fix**: Run `wildfire-rl build-tensors --region california --grid 32` to regenerate with metadata.

## Issue 8: download_data.py is a stub — no actual download logic
- **Problem**: [download_data.py](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/scripts/download_data.py#L37-L39) contains `logger.info("Install geo extras and implement the CDS request for production use.")` instead of actual ERA5/FIRMS/DEM download code. The data pipeline is not end-to-end runnable from raw sources.
- **Fix**: Implement the actual CDS/EE/FIRMS API calls (even if behind a `--dry-run` flag), or clearly document that pre-built tensors must be obtained externally.

## Issue 9: Ablation uses only seed[0], not multi-seed
- **Problem**: [run_ablation.py L52](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/scripts/run_ablation.py#L52) uses `seed = cfg.seeds[0]` — a single seed per variant. The ablation config specifies `seeds: [0, 1, 2]` but the script ignores seeds 1 and 2.
- **Fix**: Loop over all seeds per variant and report mean ± std across seeds for each ablation.

## Issue 10: Evaluation trains and evaluates on the same fixed scenario
- **Problem**: With `randomize_ignition: false` (the default in all experiment configs), every training episode and every evaluation episode uses **the identical fire channel** from the state tensor. The agent is evaluated on the exact scenario it trained on — there is no held-out test set.
- **Evidence**: [config.py L74](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/config.py#L74): `randomize_ignition: bool = False`. [reproducibility.md L50-52](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/docs/reproducibility.md#L50-L52) acknowledges this.
- **Fix**: At minimum, report results with `randomize_ignition=true` as the primary evaluation mode. Without this, claimed performance is memorization of one scenario, not generalization.

## Issue 11: MARL scaling experiment uses unequal training budgets per agent (effectively)
- **Problem**: While the config claims matched budgets ([scaling.yaml L18](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/configs/experiment/scaling.yaml#L18)), a MultiDiscrete(5^N) action space with N agents has exponentially larger action spaces. 100k timesteps for 5 agents samples the space far less densely than 100k for 1 agent. The comparison is not iso-sample-complexity.
- **Fix**: At minimum, acknowledge this confound. Better: report a sample-complexity-adjusted comparison or scale timesteps with agent count.

---

# 🟢 MINOR ISSUES

- [figures/.gitkeep](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/figures/.gitkeep): The `figures/` directory is empty (only `.gitkeep`). README line 39 says *"Architecture/result figures live in `figures/` after `make figures`"* — no figures exist.
- [report.md](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/docs/paper/report.md#L398): §26 still references `/content/drive/MyDrive/PyroRL_Saudi_Project/` — a Colab path. Should reference the repo structure.
- [report.md](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/docs/paper/report.md#L287-L289): §17 MARL scaling claims (1-agent: 0.688, 3-agent: 0.629, 5-agent: 0.513) are unverifiable — no CSV contains these numbers and no generation script produces them.
- [cnn.py](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/models/cnn.py): The `use_pooling=True` default changes the model architecture from the one that produced the legacy 200MB checkpoints. The stored checkpoints (`ppo_*_100k_seed_*.zip`, ~200MB each) were likely trained without pooling and will fail to load with the current default.
- [multi_agent.py L103](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/envs/multi_agent.py#L103): Multi-agent reward is `−total_fire` (raw, no normalization option, no per-agent bonus). Inconsistent with the single-agent env which has `reward_mode` and `suppression_bonus`.
- [environment.yml](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/environment.yml): Referenced in docs but not inspected — should be verified for pin consistency with `requirements.txt`.
- Paper figures in `docs/paper/figures/` contain two 15KB placeholder images (`agent_trajectory.png`, `wildfire_temporal_propagation.png`) that appear to be minimal placeholders, not publication-quality figures.

---

# 🔬 MISSING EXPERIMENTS

### 1. Baseline comparison (Random + NoOp vs PPO)
- **What**: Evaluate `RandomPolicy` and `NoOpPolicy` on both regions, 5 seeds × 20 episodes each
- **How**: `wildfire-rl evaluate --config configs/experiment/multiseed.yaml` (for each region)
- **Why**: Without baselines, the claim "PPO learns effective suppression" is unfalsifiable
- **Expected outcome**: PPO should achieve significantly lower total fire than Random and NoOp. If differences are <10%, the agent hasn't learned meaningful policy

### 2. Full symmetric transfer matrix
- **What**: All 4 cells: Saudi→Saudi, Saudi→CA, CA→CA, CA→Saudi
- **How**: `wildfire-rl transfer --config configs/experiment/transfer.yaml`
- **Why**: Tests whether ecological specialization is symmetric. If CA→Saudi performs differently than Saudi→CA, it reveals directional domain shift
- **Expected outcome**: Native policies outperform transferred. If Saudi→CA ≈ CA→CA, the transfer claim is falsified

### 3. Randomized-ignition evaluation (generalization)
- **What**: Train and evaluate with `randomize_ignition: true`, separate seeds for train and eval episodes
- **How**: Add `env.randomize_ignition: true` to multiseed.yaml, retrain, re-evaluate
- **Why**: Current results measure memorization of one fire scenario. Generalization requires evaluation on unseen ignition patterns
- **Expected outcome**: Slight performance drop vs fixed-scenario (expected); catastrophic drop would reveal overfitting

### 4. Statistical significance of transfer degradation
- **What**: Paired t-test (or Wilcoxon) comparing native vs transferred policy performance, per episode
- **How**: Compute per-episode metrics for native and transferred policies on same base_seeds, run scipy.stats.ttest_rel
- **Why**: The 100% reward difference claim needs a p-value
- **Expected outcome**: p < 0.05 with N=100 (5 seeds × 20 episodes). If not, the transfer claim is noise

### 5. Ablation with multi-seed + significance
- **What**: Each ablation variant (no_wind, no_terrain, no_suppression, dense_fuel) run over 3+ seeds with t-test vs baseline
- **How**: Fix [run_ablation.py](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/scripts/run_ablation.py) to loop over all seeds; add pairwise comparison
- **Why**: Single-seed ablations have no error bars; effect direction could be noise
- **Expected outcome**: no_suppression should be clearly worse than baseline (validates suppression mechanism). If not, the dynamics model is suspect

---

# 🛠️ ACTION PLAN (prioritized fix roadmap)

## Step 1 — Rebuild results from canonical pipeline (Est. effort: 4h)
- **Files**: `scripts/train.py`, `scripts/evaluate.py`, `scripts/transfer.py`, `configs/experiment/multiseed.yaml`
- **Changes**: Run `make reproduce` end-to-end for both regions. Verify that `results/runs/` metadata and `results/transfer_matrix.csv` are generated. Fix any runtime errors. Compare canonical results against legacy CSVs to check for major discrepancies.
- **Validates**: Issue 1 (untraceable results), Issue 3 (incomplete transfer matrix)

## Step 2 — Add baseline comparison to all evaluations (Est. effort: 2h)
- **Files**: `src/wildfire_rl/cli.py` (cmd_evaluate already includes baselines), evaluation configs
- **Changes**: Run `wildfire-rl evaluate` for each region, include Random and NoOp rows in publication table. Add a `results/baseline_comparison.csv`.
- **Validates**: Issue 2 (no baselines). This is the single most important experiment.

## Step 3 — Fix normalization and re-run transfer with normalized rewards (Est. effort: 2h)
- **Files**: `configs/experiment/transfer.yaml`, `data/normalize.py`
- **Changes**: Set `reward_mode: normalized` in transfer config. Optionally compute shared normalization. Report both raw and normalized results.
- **Validates**: Issue 4 (confounded cross-region comparison)

## Step 4 — Add statistical significance tests (Est. effort: 3h)
- **Files**: New `src/wildfire_rl/eval/significance.py`, updated ablation/transfer scripts
- **Changes**: Implement pairwise t-tests, 95% CIs, Cohen's d for PPO vs baselines, native vs transfer, each ablation vs baseline. Add results to CSVs.
- **Validates**: Issue 5 (zero statistical rigor)

## Step 5 — Fix ablation to use multi-seed (Est. effort: 1h)
- **Files**: `scripts/run_ablation.py`
- **Changes**: Loop over `cfg.seeds` instead of `cfg.seeds[0]`. Report per-seed and aggregate statistics.
- **Validates**: Issue 9 (single-seed ablation)

## Step 6 — Enable and evaluate randomized ignition (Est. effort: 2h)
- **Files**: `configs/experiment/multiseed.yaml`, evaluation workflow
- **Changes**: Create a `multiseed_generalization.yaml` with `randomize_ignition: true`. Train and evaluate. Compare to fixed-ignition results.
- **Validates**: Issue 10 (train/eval on same scenario)

## Step 7 — Clean up legacy artifacts and documentation (Est. effort: 1h)
- **Files**: `results/legacy_notebook_runs/`, `docs/paper/report.md`
- **Changes**: Move legacy CSVs to an archival location or clearly label them as superseded. Fix Colab path references in report.md. Regenerate figures.
- **Validates**: Minor documentation issues

---

# 📊 FINAL SCORECARD

| Dimension | Score | Key reason (one sentence) |
|---|---|---|
| Reproducibility | 6/10 | Excellent seeding/config infrastructure, but canonical pipeline has never been run end-to-end and all reported numbers come from untraceable legacy files |
| Experimental validity | 3/10 | No baselines, no held-out scenarios, per-region normalization confounds cross-region claims, train=test for single fixed ignition |
| Model/algorithm correctness | 7/10 | Clean PPO integration via SB3, correct CNN architecture, proper dynamics implementation with seeded RNG; minor multi-agent reward inconsistency |
| System design realism | 5/10 | 32×32 grid is acknowledged as toy; simplified fire physics; no latency/compute/scalability analysis; centralized MARL won't scale |
| Baselines & comparisons | 2/10 | Baselines exist in code but have never been evaluated; no SOTA RL comparison; no prior wildfire-RL method comparison |
| Code quality | 8/10 | Excellent refactor from notebooks to package; clean separation of concerns; comprehensive docstrings; OmegaConf schema; CI/lint/type checks |
| Logging & observability | 6/10 | Run metadata framework exists but has never produced output; tensorboard log path is wired but not demonstrated; no training curves available |
| Robustness | 2/10 | No distribution shift testing (beyond transfer), no adversarial/noisy inputs, no sensitivity analysis, no calibration assessment |
| Claim–implementation alignment | 4/10 | MARL scaling numbers (§17) have no backing CSV; report references Colab paths; transfer claim incomplete; baseline code exists but results don't |
| Statistical validity | 1/10 | No significance tests, no CIs, no effect sizes, no ablation statistics; single-seed ablation; 5 seeds but no inferential analysis |
| **Overall** | **5.5/10** | |

---

# ⚖️ FRAUD RISK ASSESSMENT

**Rating: LOW**

**Summary of triggered fraud fingerprints:**

| # | Fingerprint | Status | Evidence |
|---|---|---|---|
| 1 | Hardcoded metrics | **ABSENT** | No hardcoded return values found in any `.py` source file |
| 2 | Constant model outputs | **ABSENT** | Model uses standard SB3 PPO with proper CNN forward pass |
| 3 | Reverse-engineered fixtures | **ABSENT** | No suspicious test fixtures matching ground truth |
| 4 | Disconnected eval pipeline | **PRESENT** | Legacy CSVs disconnected from canonical pipeline; canonical pipeline exists but has never produced results |
| 5 | Unreachable training code | **ABSENT** | All model/training code is reachable via CLI and scripts |
| 6 | Result files without generation scripts | **PRESENT** | 8 CSVs in `legacy_notebook_runs/` with no generation script (legacy notebooks are "superseded") |
| 7 | Suspiciously round numbers | **ABSENT** | All values have realistic precision (e.g., −40525.73859374999, 521.22) |
| 8 | Seed sensitivity concealment | **ABSENT** | 5 seeds reported with std dev; seeding infrastructure is thorough |
| 9 | Test set contamination | **PRESENT** | `randomize_ignition=false` means train and eval use identical fire scenarios |
| 10 | Label leakage | **ABSENT** | RL environment — no labels passed to prediction |
| 11 | Metric implementation bugs | **ABSENT** | Metrics are straightforward sums/counts with documented thresholds |
| 12 | Fabricated dataset | **ABSENT** | Real remote-sensing sources documented; channels show realistic distributions |

**Assessment**: The combination of flags (4, 6, 9) is consistent with **innocent engineering debt from a notebook-to-package refactoring project**, not deliberate fabrication. The refactored codebase is remarkably transparent about its limitations — the docstrings and comments explicitly flag the issues I identified (missing baselines in original, hardcoded transfer rows, inconsistent thresholds). The author appears to have built the correct infrastructure to address these problems but has not yet run the full pipeline to produce verified canonical results. This is engineering incompleteness, not fraud.

---

# Phase 0A — REPOSITORY MAP

```
wildfire-rl/
├── .dockerignore                    # Docker ignore rules                                    [used by Dockerfile]
├── .env.example                     # Env var template for credentials                       [referenced in README/download_data.py]
├── .github/workflows/ci.yml         # CI: lint, test, install, large-file guard              [active CI]
├── .gitattributes                   # Git LFS/attribute config                               [active]
├── .gitignore                       # Ignore rules                                           [active]
├── .pre-commit-config.yaml          # Pre-commit hooks                                       [used by make install-dev]
├── CHANGELOG.md                     # Version changelog                                      [standalone]
├── CITATION.cff                     # Citation metadata                                      [referenced in README]
├── CODE_OF_CONDUCT.md               # Community standards                                    [standalone]
├── CONTRIBUTING.md                  # Contribution guidelines                                [standalone]
├── Dockerfile                       # Reproducible CPU runtime                               [standalone]
├── LICENSE                          # MIT license                                             [standalone]
├── Makefile                         # Build/workflow entrypoints                              [primary automation]
├── README.md                        # Main documentation                                     [primary docs]
├── pyproject.toml                   # Package metadata + deps + tool config                  [used by pip/setuptools]
├── requirements.txt                 # Pinned runtime deps                                    [used by Dockerfile/pip]
├── requirements-dev.txt             # Dev deps (thin)                                        [standalone]
├── requirements-geo.txt             # Geospatial stack deps                                  [standalone]
├── environment.yml                  # Conda env spec                                         [referenced in docs]
│
├── configs/
│   ├── config.yaml                  # Base config (documents defaults)                       [loaded by config.py]
│   ├── env/single.yaml              # Single-agent env dynamics                              [composable overlay]
│   ├── env/multi.yaml               # Multi-agent env config                                 [composable overlay]
│   ├── ppo/default.yaml             # PPO hyperparameters                                    [composable overlay]
│   ├── region/california.yaml       # CA region spec                                         [composable overlay]
│   ├── region/saudi.yaml            # Saudi region spec                                      [composable overlay]
│   ├── experiment/multiseed.yaml    # Multi-seed training config                             [used by train/evaluate]
│   ├── experiment/transfer.yaml     # Transfer matrix config                                 [used by transfer.py]
│   ├── experiment/ablation.yaml     # Ablation study config                                  [used by run_ablation.py]
│   └── experiment/scaling.yaml      # MARL scaling config                                    [used by train_marl.py]
│
├── src/wildfire_rl/
│   ├── __init__.py                  # Package init (version)                                 [active]
│   ├── cli.py                       # CLI entrypoint (train/evaluate/transfer/info)          [active]
│   ├── config.py                    # OmegaConf config schema + loader                       [imported everywhere]
│   ├── paths.py                     # Centralized path resolution                            [imported by CLI/scripts]
│   ├── seeding.py                   # Global + per-env RNG seeding                           [imported by train/tests]
│   ├── logging_utils.py             # Logger + run metadata                                  [imported by CLI/scripts]
│   ├── py.typed                     # PEP 561 marker                                         [standalone]
│   ├── data/__init__.py             # Data package exports                                   [active]
│   ├── data/normalize.py            # Min-max normalization                                  [used by tensor_stack, tests]
│   ├── data/tensor_stack.py         # Channel stacking → state_tensor.npy                   [used by CLI/tests]
│   ├── data/manifest.py             # SHA256 integrity manifests                             [used by fetch_models/make_manifest]
│   ├── envs/__init__.py             # Env package exports                                    [active]
│   ├── envs/base.py                 # WildfireEnv (single-agent)                             [used by train/eval/tests]
│   ├── envs/dynamics.py             # Pure fire spread/suppression/decay functions           [used by base.py/multi_agent.py]
│   ├── envs/multi_agent.py          # MultiAgentWildfireEnv                                  [used by train_marl.py]
│   ├── models/__init__.py           # Model package exports                                  [active]
│   ├── models/cnn.py                # CustomCNN feature extractor                            [used by train/ppo.py]
│   ├── train/__init__.py            # Train package exports                                  [active]
│   ├── train/ppo.py                 # PPO build + train functions                            [used by CLI/scripts]
│   ├── eval/__init__.py             # Eval package exports                                   [active]
│   ├── eval/baselines.py            # RandomPolicy + NoOpPolicy                              [used by CLI/tests]
│   ├── eval/evaluate.py             # Canonical evaluation loop                              [used by CLI/transfer/tests]
│   ├── eval/metrics.py              # burned_cells, fire_intensity, summarize                [used by evaluate.py/tests]
│   ├── eval/transfer.py             # N×N transfer matrix computation                       [used by transfer_run.py]
│   ├── experiments/__init__.py      # Experiments package                                    [active]
│   ├── experiments/transfer_run.py  # Transfer orchestration from config                     [used by CLI]
│   ├── viz/__init__.py              # Viz package exports                                    [active]
│   └── viz/figures.py               # Deterministic figure generation                        [used by CLI make-figures]
│
├── scripts/
│   ├── _bootstrap.py                # sys.path setup for source checkouts                    [imported by all scripts]
│   ├── train.py                     # Train CLI wrapper                                      [used by Makefile]
│   ├── evaluate.py                  # Evaluate CLI wrapper                                   [used by Makefile]
│   ├── transfer.py                  # Transfer CLI wrapper                                   [used by Makefile]
│   ├── run_ablation.py              # Ablation study runner                                  [used by Makefile]
│   ├── train_marl.py                # MARL scaling experiment                                [used by Makefile]
│   ├── make_figures.py              # Figure generation wrapper                              [used by Makefile]
│   ├── download_data.py             # Raw data download (**STUB** — no actual download logic) [used by Makefile]
│   ├── build_tensors.py             # Tensor building wrapper                                [used by Makefile]
│   ├── make_manifest.py             # Checksum manifest generator                            [used by Makefile]
│   ├── fetch_models.py              # HF Hub model downloader                                [standalone]
│   ├── check_repo_size.py           # Large-file guard                                       [used by Makefile/CI]
│   └── strip_notebooks.py           # Notebook output stripper                               [used by Makefile]
│
├── tests/
│   ├── conftest.py                  # Shared fixtures (small_tensor, env_cfg)                [used by all tests]
│   ├── test_env_api.py              # Env shapes, truncation, determinism, Gymnasium check   [active]
│   ├── test_metrics.py              # Metric threshold, fire_intensity, summarize            [active]
│   ├── test_seeding.py              # Global seed + isolated RNG reproducibility             [active]
│   ├── test_eval_baselines.py       # Baseline policies + evaluation loop                    [active]
│   ├── test_config.py               # Config loading, overrides, serialization               [active]
│   ├── test_tensor_stack.py         # Tensor stacking, normalization, shared scaler          [active]
│   └── test_cli_smoke.py            # CLI info, version, subcommand requirement              [active]
│
├── data/
│   ├── README.md                    # Data provenance docs                                   [standalone]
│   ├── sample/state_tensor.npy      # Synthetic 7×8×8 test tensor (committed to git)         [used by tests]
│   ├── sample/README.md             # Sample data docs                                       [standalone]
│   ├── california/grids/32x32/      # Real CA tensors (7 .npy channels + state_tensor.npy)   [used by training]
│   ├── california/grids/64x64/      # Higher-res CA tensors                                  [future use]
│   ├── california/raw/{dem,era5,firms,ndvi}/ # Raw CA data directories (empty)              [placeholder]
│   ├── california/processed/        # Processed CA data (empty)                              [placeholder]
│   ├── saudi_eastern_province/grids/32x32/ # Real Saudi tensors                             [used by training]
│   ├── saudi_eastern_province/grids/64x64/ # Higher-res Saudi tensors                      [future use]
│   ├── saudi_eastern_province/raw/  # Raw Saudi data directories (empty+osm)                [placeholder]
│   └── saudi_eastern_province/processed/ # Processed Saudi data                             [placeholder]
│
├── models/                          # PPO checkpoints (~200MB each, 12 total ≈ 2.4GB)        [used by eval/transfer]
│   ├── README.md
│   └── ppo_*_{seed}.zip             # 12 checkpoint files + 1 legacy checkpoint
│
├── results/
│   ├── README.md                    # Results provenance docs                                [standalone]
│   └── legacy_notebook_runs/        # ⚠️ 8 CSVs from superseded notebooks (no generation script)
│       ├── cross_region_evaluation.csv
│       ├── final_summary_statistics.csv
│       ├── multi_seed_results_california.csv
│       ├── multi_seed_results_saudi.csv
│       ├── multi_seed_statistics.csv
│       ├── phase6_ablation_results.csv
│       ├── publication_results_table.csv
│       └── zero_shot_transfer_results.csv
│
├── notebooks/
│   ├── README.md                    # Notebook docs                                          [standalone]
│   ├── 01_quickstart_demo.ipynb     # Quickstart demo                                        [standalone]
│   └── legacy/                      # Original notebooks (Saudi/, California/, analysis)     [archival]
│
├── docs/
│   ├── architecture.md              # Package architecture                                   [standalone]
│   ├── reproducibility.md           # Reproducibility protocol                               [standalone]
│   ├── data_card.md                 # Dataset documentation                                  [standalone]
│   ├── model_card.md                # Model documentation                                    [standalone]
│   ├── development.md               # Dev guide                                              [standalone]
│   ├── REPOSITORY_AUDIT_AND_PLAN.md # Prior audit document                                   [archival]
│   ├── AUDIT_V2_FORENSIC_CODE_REVIEW.md # Prior forensic review                             [archival]
│   └── paper/
│       ├── report.md                # Comprehensive project report                            [primary paper]
│       └── figures/                 # 4 paper figures (2 placeholders, 2 real)                [used by report]
│
└── figures/                         # ⚠️ Empty (only .gitkeep) — no generated figures exist
    └── .gitkeep
```

**Flags:**
- ⚠️ 8 CSV result files in `legacy_notebook_runs/` have NO generation script in the current codebase
- ⚠️ `figures/` is empty despite README claiming figures are generated there
- ⚠️ `download_data.py` is a stub (no actual download logic)
- ⚠️ MARL scaling numbers in report §17 have no corresponding result file

---

# Phase 0B — CLAIM LEDGER

| # | Claim | Value | Source file | Verified? | Evidence or gap |
|---|---|---|---|---|---|
| 1 | Saudi mean episode reward | −40525.74 ± 772.68 | publication_results_table.csv | ❌ No | Legacy CSV; no run metadata; no generation script |
| 2 | California mean episode reward | −86180.23 ± 1277.77 | publication_results_table.csv | ❌ No | Same as above |
| 3 | Saudi mean burned cells | 521.22 ± 6.75 | publication_results_table.csv | ❌ No | Same as above |
| 4 | California mean burned cells | 794.10 ± 3.20 | publication_results_table.csv | ❌ No | Same as above |
| 5 | Saudi mean fire intensity | 472.09 ± 5.73 | publication_results_table.csv | ❌ No | Same as above |
| 6 | California mean fire intensity | 713.15 ± 2.50 | publication_results_table.csv | ❌ No | Same as above |
| 7 | Transfer: Saudi→Saudi reward | −41284.78 | cross_region_evaluation.csv | ❌ No | Legacy; different from multi-seed table |
| 8 | Transfer: Saudi→CA reward | −86674.52 | cross_region_evaluation.csv | ❌ No | Legacy; only 1 seed, no error bar |
| 9 | Transfer: CA→CA reward | −85326.05 | cross_region_evaluation.csv | ❌ No | Legacy; missing CA→Saudi cell |
| 10 | Transfer degradation >100% | Implied | report.md §22 | ❌ No | Confounded by normalization |
| 11 | 5 seeds used | 5 | multi_seed_results_*.csv | ✅ Partial | CSVs contain 5 rows; but no run metadata proves these are independent runs |
| 12 | 1-agent fire: 0.688 | 0.688 | report.md §17 | ❌ No | No backing CSV anywhere |
| 13 | 3-agent fire: 0.629 | 0.629 | report.md §17 | ❌ No | No backing CSV anywhere |
| 14 | 5-agent fire: 0.513 | 0.513 | report.md §17 | ❌ No | No backing CSV anywhere |
| 15 | ~25% improvement with more agents | ~25% | report.md §17 | ❌ No | Derived from unverifiable MARL numbers |
| 16 | Ablation: baseline fire 0.303 | 0.303 | phase6_ablation_results.csv | ❌ No | Legacy; single seed |
| 17 | Ablation: no_wind fire 0.232 | 0.232 | phase6_ablation_results.csv | ❌ No | Legacy; single seed |
| 18 | Ablation: dense_fuel fire 0.625 | 0.625 | phase6_ablation_results.csv | ❌ No | Legacy; single seed |
| 19 | 7-channel state tensor | 7 channels | config.py, tensor_stack.py | ✅ Yes | Verified: tensors are (7,32,32) |
| 20 | 32×32 grid size | 32 | config.py, data/ | ✅ Yes | Verified: tensors are 32×32 |
| 21 | 100k timesteps training | 100000 | configs/ppo/default.yaml | ✅ Yes | Config value matches PPOConfig default |

**Summary**: 15 of 21 quantitative claims (71%) are **unverifiable** — they trace only to legacy CSVs with no generation script or run metadata. This exceeds the 30% threshold.

> [!CAUTION]
> Over 70% of quantitative claims cannot be traced to live computation in the current codebase. While the claims appear plausible and the legacy CSV values have realistic precision (arguing against fabrication), they are formally untraceable.

---

# Phase 1 — FRAUD FINGERPRINT AUDIT

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Hardcoded metrics | **ABSENT** | grep found no `return 0.XX`, `accuracy = 0.XX`, or `assert metric == X` in src/ |
| 2 | Constant model outputs | **ABSENT** | SB3 PPO with proper CNN forward pass; actions vary with observation |
| 3 | Reverse-engineered fixtures | **ABSENT** | Test fixtures are synthetic (conftest.py) with no suspiciously matching values |
| 4 | Disconnected eval pipeline | **PRESENT** | Legacy CSVs ≠ canonical pipeline. The canonical pipeline exists and is properly wired, but has never produced output. |
| 5 | Unreachable training code | **ABSENT** | All model/train/eval code is reachable via CLI entry points |
| 6 | Result files without generation scripts | **PRESENT** | 8 CSVs in legacy_notebook_runs/ with no generation script |
| 7 | Suspiciously round numbers | **ABSENT** | Values like −40525.73859374999 and 521.22 are realistically precise |
| 8 | Seed sensitivity concealment | **ABSENT** | 5 seeds with std dev; seeding is thorough (global + per-env) |
| 9 | Test set contamination | **PRESENT** | `randomize_ignition=false` → train and eval on identical scenario. Documented as known limitation. |
| 10 | Label leakage | **ABSENT** | RL setup — no labels passed to prediction |
| 11 | Metric implementation bugs | **ABSENT** | Metrics are simple sums/counts; threshold documented and configurable |
| 12 | Fabricated dataset | **ABSENT** | Real remote-sensing sources; channel distributions look realistic (per numpy inspection) |

---

# Phase 2 — PIPELINE RECONSTRUCTION

```
[raw data]              →  [preprocessing]           →  [state tensor]           →  [training]              →  [evaluation]            →  [metrics]
download_data.py (STUB)    normalize.py/tensor_stack.py  data/<region>/grids/       train/ppo.py via CLI       eval/evaluate.py           eval/metrics.py
                           RUNNABLE                      32x32/state_tensor.npy     RUNNABLE (needs tensor)    RUNNABLE (needs model)     RUNNABLE
                                                         EXISTS for both regions
```

**Q1. Can a reviewer clone this repo and reproduce the main result in < 2 hours?**
No. The tensors are present (~29-57KB), but the 12 model checkpoints total ~2.4GB and must be fetched from HF Hub (which requires `fetch_models.py` + HF credentials). Training from scratch at 100k timesteps × 5 seeds × 2 regions would take many hours on CPU. The `make reproduce` target exists but has never been tested end-to-end.

**Q2. Are random seeds fixed at every stochastic operation?**
Yes — above average for an RL project. `set_global_seed()` covers Python, NumPy, PyTorch, CUDA, and SB3. The environment uses `self.np_random` (seeded via Gymnasium's `reset(seed=...)`) not the global `np.random`. Tests verify determinism (`test_determinism_same_seed`).

**Q3. Is the train/val/test split performed before any feature engineering?**
N/A in the traditional sense (RL, not supervised learning). However, with `randomize_ignition=false`, there is no train/test split at all — identical scenario for training and evaluation. This is the most significant experimental validity issue.

**Q4. Is the evaluation script independent from the training script?**
Yes. `eval/evaluate.py` creates a fresh env via factory, resets with explicit seeds, and runs episodes independently. It accepts any `Policy` object (SB3 model, Random, NoOp). Training and evaluation are cleanly separated.

**Q5. Do reported metrics match what the code would actually compute?**
Uncertain. The legacy CSVs used inconsistent threshold definitions (docstring in metrics.py L3-5 notes notebooks used threshold > 0.5 in training but > 0.2 in evaluation). The canonical code now uses 0.5 consistently, but the legacy numbers may have been computed with the inconsistent thresholds.

---

# Phase 3 — STATISTICAL VALIDITY GATE

| Check | Status |
|---|---|
| Results over ≥ 3 independent seeds | ✅ 5 seeds per region |
| Std dev reported | ✅ Present in all CSVs |
| Confidence intervals | ❌ **ABSENT** |
| Statistical significance test vs each baseline | ❌ **ABSENT** (no baselines evaluated) |
| Effect size (Cohen's d) | ❌ **ABSENT** |
| Ablation table with significance | ❌ **ABSENT** (single seed, no test) |
| No cherry-picking | ✅ All 5 seeds reported |

> [!CAUTION]
> **CRITICAL**: All 6 statistical validity items except seed count and std dev are absent. This is the most significant gap for publishability.

---

# Phase 4 — CROSS-FILE CONSISTENCY MATRIX

| Pair | Status | Evidence |
|---|---|---|
| README metrics ↔ eval script | **UNCERTAIN** | README quotes no specific numbers; metrics names (episode_reward, burned_cells, fire_intensity) match eval/metrics.py. Threshold = 0.5 consistent. But README points to `make reproduce` which has never run. |
| Config hyperparams ↔ train script | **CONSISTENT** | All PPOConfig fields are loaded via OmegaConf and passed to SB3 PPO constructor (train/ppo.py L30-47). Verified: learning_rate, n_steps, batch_size, gamma, etc. all flow correctly. |
| Paper architecture ↔ model code | **CONSISTENT** | Report §12.3 describes Conv2D+ReLU+Flatten+Linear. cnn.py implements exactly this. The pooling addition is new but documented. |
| Dataset description ↔ data loader | **CONSISTENT** | 7 channels, 32×32, same order in config.py, tensor_stack.py, data_card.md, and architecture.md. Verified tensors match (7, 32, 32). |
| Requirements.txt ↔ imports | **CONSISTENT** | All imports (numpy, pandas, matplotlib, gymnasium, stable-baselines3, torch, omegaconf, pyyaml, tqdm) have pinned versions in requirements.txt. No missing dependency. |

---

# Phase 6 — SILENT BUG HUNT

| # | Anti-pattern | Status | Evidence |
|---|---|---|---|
| 1 | Off-by-one in sequence modeling | **NOT FOUND** | Not a sequence model — grid-based RL env |
| 2 | sklearn Pipeline fit on full dataset | **NOT FOUND** | No sklearn used; normalization is per-channel min-max on raw data, not on train/test splits |
| 3 | DataFrame index leakage | **NOT FOUND** | No DataFrame-based data loading for model input |
| 4 | Metrics on training set labeled as "test" | **UNCERTAIN** | With `randomize_ignition=false`, train and eval use the same scenario, so metrics are effectively on the training distribution, though evaluated via separate rollouts |
| 5 | Class weight not applied but claimed | **NOT FOUND** | No class weights (RL, not classification) |
| 6 | Softmax applied twice | **NOT FOUND** | No softmax in cnn.py; SB3 PPO handles action distribution internally |
| 7 | Data augmentation on val/test | **NOT FOUND** | No data augmentation anywhere |
| 8 | LR scheduler called at wrong frequency | **NOT FOUND** | No LR scheduler (constant LR via SB3 default) |
