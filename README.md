<div align="center">

# 🔥 wildfire-marl

**Hierarchical multi-agent reinforcement learning for wildfire suppression,<br/>evaluated on validated fire physics.**

[**English**](README.md) · [**العربية**](README.ar.md)

[![CI](https://github.com/aliakarma/wildfire-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/aliakarma/wildfire-rl/actions/workflows/ci.yml)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22C55E.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Simulator: Cell2Fire](https://img.shields.io/badge/simulator-Cell2Fire-EA580C.svg)](third_party/README.md)
[![Frozen artifacts: SHA-256](https://img.shields.io/badge/artifacts-SHA--256%20frozen-6366F1.svg)](results/README.md)
[![Verification: 42 checks](https://img.shields.io/badge/verification-42%20checks%20passing-16A34A.svg)](docs/supplement_assets/REPRODUCIBILITY.md)
[![Dashboard: bilingual](https://img.shields.io/badge/dashboard-EN%20%2F%20AR%20RTL-0EA5E9.svg)](dashboard/README.md)

</div>

---

A team of firefighting agents defends critical infrastructure against fire spreading under
**Cell2Fire** — a peer-reviewed C++ simulator implementing the Canadian Forest Fire Behaviour
Prediction (FBP) system. The landscapes are real: 7-channel geospatial tensors (ERA5 weather,
SRTM terrain, MODIS NDVI fuel, FIRMS ignitions) for two deliberately dissimilar regimes —
**Saudi Arabia's Eastern Province** (desert / petroleum) and **Northern California** (forest /
wildland-urban interface).

Cell2Fire's physics is never modified. All suppression and coordination logic lives in the RL
wrapper, so the fire model stays exactly as published.

---

## 📋 Table of Contents

- [🎬 Watch it run](#-watch-it-run)
- [🧭 Overview](#-overview)
  - [What this repository provides](#what-this-repository-provides)
  - [Metrics](#metrics)
- [📊 Headline results](#-headline-results)
  - [Ablations: what the data says](#ablations-what-the-data-says)
- [🧠 Method](#-method)
- [✅ Requirements](#-requirements)
- [⚙️ Installation](#️-installation)
- [🚀 Quick start](#-quick-start)
- [🔬 Reproducing the paper](#-reproducing-the-paper)
- [🎛️ Configuration](#️-configuration)
- [🗂️ Repository layout](#️-repository-layout)
- [📈 Results dashboard](#-results-dashboard)
- [🔒 Reproducibility guarantees](#-reproducibility-guarantees)
- [🛠️ Development](#️-development)
- [🩺 Troubleshooting](#-troubleshooting)
- [⚠️ Limitations](#️-limitations)
- [📚 Provenance](#-provenance)
- [🤝 Contributing](#-contributing)
- [📝 Citation](#-citation)
- [⚖️ License](#️-license)

---

## 🎬 Watch it run

Single held-out episodes rendered from the frozen rollouts. Left: **No-Op**, the do-nothing
control. Right: **HierComm**, the proposed policy. Aggregate numbers are in
[Headline results](#-headline-results) — these clips are illustrations, not measurements.

<table>
<tr>
<th width="50%">🚫 No-Op — no suppression</th>
<th width="50%">🧯 HierComm (ours)</th>
</tr>
<tr>
<td><img src="docs/media/rollout_noop_saudi.gif" alt="No-Op rollout, Saudi Eastern Province" width="100%"/></td>
<td><img src="docs/media/rollout_hiercomm_saudi.gif" alt="HierComm rollout, Saudi Eastern Province" width="100%"/></td>
</tr>
<tr>
<td colspan="2" align="center"><i>Saudi Arabia — Eastern Province (desert / petroleum)</i></td>
</tr>
<tr>
<td><img src="docs/media/rollout_noop_california.gif" alt="No-Op rollout, Northern California" width="100%"/></td>
<td><img src="docs/media/rollout_hiercomm_california.gif" alt="HierComm rollout, Northern California" width="100%"/></td>
</tr>
<tr>
<td colspan="2" align="center"><i>Northern California (forest / wildland-urban interface)</i></td>
</tr>
</table>

> [!TIP]
> Full-resolution rollouts for **every** policy and region — plus cross-region transfer
> replays — are published as MP4 under
> [`dashboard/public/media/`](dashboard/public/media/wildfire_phase5_gifs) and are browsable
> interactively in the [results dashboard](#-results-dashboard). Media is regenerated with
> `scripts/render_phase5_gifs.py` and `scripts/render_phase6_transfer.py`.

---

## 🧭 Overview

Wildfire suppression is a coordination problem: a small crew must decide *which* assets to
defend and *where* to cut line, under a fire whose spread they do not control. This repository
studies that problem as a cooperative multi-agent RL task on top of a validated fire simulator,
rather than a bespoke fire model tuned alongside the policy.

### What this repository provides

| | |
|---|---|
| 🗺️ **A benchmark** | Two dissimilar real landscapes, a suppression-relevant scenario regime, and an evaluation protocol with disjoint train/eval ignition streams. |
| 📐 **Baselines** | Four information-matched heuristics and three learned methods (MAPPO, CommNet, and QMIX — the last reported as a failed baseline). |
| 🧠 **A method** | HierComm, a two-level policy combining value-aware asset dispatch with a learned tactical layer. See [Method](#-method). |
| 🔒 **Frozen artifacts** | Every reported number derives from committed per-episode CSVs, each covered by a SHA-256 manifest, and is re-derivable in seconds by one command. |

### Metrics

Two metrics carry the headline results. Both are defined once, in
[`src/wildfire_marl/eval/metrics.py`](src/wildfire_marl/eval/metrics.py), and every caller uses
those definitions.

| Metric | Direction | Definition |
|---|:---:|---|
| **WEL** — Weighted Economic Loss | 🔻 **lower is better** | Sum over asset cells of `value[asset_type] × final fire intensity`. |
| **ISR** — Infrastructure Survival Rate | 🔺 **higher is better** | Fraction of asset cells never reached by fire (final intensity ≤ `REACH_THRESHOLD`, 0.1). |

> [!NOTE]
> **WEL is a dimensionless index, not a currency amount.** The asset values are stylized
> relative multipliers (default `{1: 10.0, 2: 4.0, 3: 6.0, 4: 3.0}`), and the underlying
> criticality raster is a `[0, 1]` Gaussian falloff around public infrastructure sites — not
> measured economic valuations. WEL is comparable *across policies on the same landscape*; it
> is not a monetary quantity. See [`docs/data_card.md`](docs/data_card.md).

Additional metrics (coordination efficiency, catastrophe prevention, risk-adjusted containment,
transfer retention) are defined in the same module and reported in the frozen artifacts.

---

## 📊 Headline results

Seven policies, two regions, five training seeds, evaluated on held-out ignition seeds
(15 episodes per seed group). Heuristics are deterministic, so they carry no seed variance.

| Method | Saudi WEL 🔻 | Saudi ISR 🔺 | California WEL 🔻 | California ISR 🔺 |
|---|---|---|---|---|
| No-Op | 27.00 | 0.286 | 34.00 | 0.000 |
| Value-First heuristic | 18.84 | 0.545 | 28.40 | 0.156 |
| Greedy-Risk heuristic | 24.72 | 0.358 | 28.40 | 0.156 |
| Local Reactive heuristic | 11.91 | 0.720 | 10.23 | 0.702 |
| Flat MARL (MAPPO) | 9.77 ± 5.10 | 0.692 | 14.17 ± 2.89 | 0.624 |
| CommNet | 12.44 ± 4.32 | 0.682 | 19.58 ± 7.77 | 0.398 |
| 🏆 **HierComm (ours)** | **7.02 ± 2.36** | **0.809** | **5.88 ± 1.04** | **0.832** |

HierComm is best in both regions and by far the most *reliable* — the lowest across-seed
variance of any learned method. On California the margin is statistically decisive (vs MAPPO
p = 0.0018, d = −3.81). On Saudi it beats every baseline except MAPPO, which it ties on the
mean (p = 0.32, n.s.) while being far less seed-sensitive.

### Ablations: what the data says

The single-factor ablations do **not** support the intuitive story, and they are reported as
measured (Saudi; five training seeds unless noted).

| Removed component | Saudi WEL 🔻 | vs. full system |
|---|---|---|
| — (full system) | 7.02 ± 2.36 | — |
| Communication | 5.67 ± 1.16 | p = 0.30 · ⚪ **not significant** |
| Tactical reward shaping | 5.41 ± 1.43 | p = 0.23 · ⚪ **not significant** |
| Hierarchy | 12.44 ± 4.32 | p = 0.048 · 🔴 significantly worse |
| RL fine-tuning (BC only) | 11.33 ± 1.26 | p = 0.011 · 🔴 significantly worse |
| Learned tactical layer <sup>†</sup> | 18.76 | 🔴 large degradation |

<sup>†</sup> Single seed (the variant is deterministic), so no significance test is reported.

Removing communication or reward shaping *lowers* WEL slightly, but neither change is
statistically significant. **Inter-agent communication is not the driver at the three-agent
scale studied.** The hierarchy, the learned tactical layer, and RL fine-tuning are. A separately
trained *learned* commander also never significantly improves on the value-aware dispatch rule,
which is why the deployed system keeps the rule.

Full protocol, significance tests, and documented reproducibility caveats:
[`docs/RESULTS_FROZEN.md`](docs/RESULTS_FROZEN.md).

---

## 🧠 Method

HierComm is a two-level cooperative policy over `N` agents (default `N = 3`), implemented in
[`src/wildfire_marl/train/hier_comm_train.py`](src/wildfire_marl/train/hier_comm_train.py).

**1. Strategic commander.** Every `macro_interval` steps, it assigns each agent a high-value
*threatened* asset to defend. Two modes exist:

- `heuristic` — a value-aware dispatch rule. **This is the configuration reported as
  "HierComm (ours)"** (method id `hiercomm_heur`); it isolates the tactical-learning
  contribution.
- `learned` — a [`StrategicController`](src/wildfire_marl/agents/strategic_controller.py)
  network, BC-warmstarted then updated by REINFORCE with a KL anchor to the BC prior. It is
  evaluated as an ablation and never significantly improves on the rule.

**2. Tactical actor.** A learned, communicating policy defends the assigned asset, conditioned
on the asset target and on mean-pooled messages from teammates. It is trained in two stages:

- **BC warm-start** — clone value-aware asset defense, which already cuts WEL well below the
  value-blind Local Reactive baseline.
- **PPO fine-tuning** — optimize the WEL/ISR objective directly (matched to evaluation) with a
  centralized critic and light firebreak shaping.

Parameters are shared across agents, so the policy is agent-count agnostic.

---

## ✅ Requirements

Requirements differ sharply by workflow. **Only the simulator itself is Linux-only.**

| Workflow | OS | Python | Additional requirements |
|---|---|:---:|---|
| 🔍 Verify published numbers | Linux · macOS · Windows | 3.10 / 3.11 | none beyond core deps |
| 🧪 Run the test suite | Linux · macOS · Windows | 3.10 / 3.11 | `[dev]` extra |
| 🔥 Run / step the simulator | **Linux only** (WSL2, native, Colab) | 3.10 / 3.11 | `g++`, Boost headers, Eigen3 |
| 🏋️ Train or reproduce experiments | **Linux only** | 3.10 / 3.11 | built simulator + PyTorch |
| 📈 Build the results dashboard | Linux · macOS · Windows | 3.10 / 3.11 | Node.js 22 (CI-pinned) |

> [!IMPORTANT]
> Cell2Fire is a C++ build and **native Windows is not supported for the simulator**. Use WSL2,
> a Linux machine, or Colab for anything that steps the environment. Verification, the test
> suite, and the dashboard build run on Windows and macOS as well — the tests that drive the
> compiled binary skip automatically.

---

## ⚙️ Installation

### 1. Clone

```bash
git clone https://github.com/aliakarma/wildfire-rl.git
cd wildfire-rl
```

The Cell2Fire fork is vendored as plain files, not a git submodule, so no `--recursive` flag
and no submodule initialization is needed.

### 2. Python environment

Conda is recommended on Linux because it supplies the C++ toolchain and the GDAL binary stack
in one step:

```bash
conda env create -f environment-linux.yml
conda activate wildfire-marl
```

Plain pip works on any platform for the verification, test, and dashboard workflows:

```bash
pip install -e ".[dev]"
```

> [!IMPORTANT]
> The `[dev]` extra does **not** include PyTorch. Every training and evaluation driver
> (`run_phase3.py`, `run_phase4_ablations.py`, `run_phase6_transfer.py`) imports `torch` at
> module load, so install it before [reproducing the paper](#-reproducing-the-paper):
>
> ```bash
> pip install -e ".[dev]" "torch>=2.1,<3.0"
> ```
>
> The optional `[baselines]` extra adds Stable-Baselines3 and sb3-contrib, used only by the
> single-agent Maskable-PPO baseline in
> [`agents/ppo_baseline.py`](src/wildfire_marl/agents/ppo_baseline.py).

### 3. Build the fire simulator (Linux only)

Required only to run the environment. Skip it if you are verifying published numbers.

```bash
sudo apt-get update
sudo apt-get install -y build-essential libboost-all-dev libeigen3-dev

cd third_party/firehose/cell2fire/Cell2FireC
make
```

This produces the `Cell2Fire` binary in that directory, which is exactly where
[`env/cell2fire_binding.py`](src/wildfire_marl/env/cell2fire_binding.py) expects it. Full
instructions and troubleshooting:
[`docs/supplement_assets/BUILD.md`](docs/supplement_assets/BUILD.md).

Confirm the binary is discoverable:

```bash
python -c "from wildfire_marl.env.cell2fire_binding import DEFAULT_BINARY; print('binary present:', DEFAULT_BINARY.exists())"
```

### 4. Optional — data-acquisition credentials

Only needed to re-run the geospatial acquisition notebooks under `notebooks/`. The generated
Cell2Fire landscapes are already committed, so experiments do not require these.

```bash
cp .env.example .env   # then fill in CDSAPI_KEY, FIRMS_MAP_KEY, EE_PROJECT_ID
```

`.env` is git-ignored. Never commit real credentials.

---

## 🚀 Quick start

### 🔍 Verify the published numbers — seconds, no GPU, no simulator build

This is the fastest way to audit the repository. It recomputes the main table, the significance
tests, the transfer matrix, and the difficulty sweep straight from the committed per-episode
CSVs, and checks each shipped file against its frozen SHA-256.

```bash
python scripts/verify_supplement.py
```

Expected final lines:

```text
42 passed, 0 failed, 0 skipped
VERIFICATION PASSED — every claim checked re-derives from the frozen artifacts.
```

Exit code is 0 only if every check passes.

### 🧪 Run the test suite

```bash
pytest tests/
```

Expected without the simulator built: **61 passed, 7 skipped**. The 7 skips are the tests that
drive the compiled Cell2Fire binary; they run once you complete the build.

### 🐍 Use the metrics directly

A minimal, runnable example — no simulator, no GPU, no PyTorch:

```python
import numpy as np
from wildfire_marl.eval.metrics import (
    infrastructure_survival_rate,
    weighted_economic_loss,
)

# Toy 4x4 landscape: asset codes (0 = no asset) and final fire intensity.
asset_type = np.array([[1, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 2, 0],
                       [0, 0, 0, 0]])
final_fire = np.zeros((4, 4))
final_fire[0, 0] = 0.9          # the type-1 asset was reached by fire
asset_values = {1: 10.0, 2: 4.0}

print("WEL:", weighted_economic_loss(asset_type, final_fire, asset_values))
print("ISR:", infrastructure_survival_rate(asset_type, final_fire))
```

```text
WEL: 9.0
ISR: 0.5
```

### 🔥 Run the simulator — Linux, after the build

```bash
python scripts/sim_smoke.py --region saudi        # or: --region california
python scripts/profile_env.py                     # step/reset throughput
```

`sim_smoke.py` runs a free-burning fire on a region landscape and checks the physics is
*plausible*, not merely runnable: monotone fire growth, downwind spread drift matched against
the mean weather wind direction, and low-fuel self-extinguish.

---

## 🔬 Reproducing the paper

> [!WARNING]
> **Checkpoints are not tracked by git** — they ship with the supplement. The drivers are
> idempotent and will **train** any checkpoint they cannot find, so an incomplete checkpoint
> directory silently turns a minutes-long evaluation into a multi-hour training job. Confirm
> the checkpoints are in place under `results/wildfire_phase3_multiseed/` first, or restrict
> the run to the seeds you actually have.

### Reproducibility tiers

Ordered by cost. [`docs/supplement_assets/REPRODUCIBILITY.md`](docs/supplement_assets/REPRODUCIBILITY.md)
maps every claim to its command and expected number.

| Tier | What it does | Cost | Needs simulator? |
|:---:|---|---|:---:|
| 1️⃣ | Re-derive every reported number from frozen CSVs | seconds | ❌ |
| 2️⃣ | Run the test suite | seconds | ❌ |
| 3️⃣ | Re-evaluate a shipped checkpoint against the real simulator | 10–30 min | ✅ |
| 4️⃣ | Full retraining from scratch | GPU-hours | ✅ |

### Full experiment commands

With checkpoints in place under `results/wildfire_phase3_multiseed/`, every study re-runs from
the CLI:

```bash
# Main results — trains 7 policies × 2 regions × 5 seeds, then evaluates on held-out seeds.
python scripts/run_phase3.py \
  --methods noop,value_first,greedy_risk,local_reactive,mappo,commnet,hiercomm_heur \
  --regions saudi,california --train-seeds 42,1042,2042,3042,4042 --parallel 6 \
  --train-steps 100000 --episodes 15 --out results/wildfire_phase3_multiseed

# Ablations and the difficulty sweep.
python scripts/run_phase4_ablations.py --study ablation \
  --ckpt-dir results/wildfire_phase3_multiseed --out /tmp/ablations

# Saudi <-> California transfer matrix and held-out generalization.
python scripts/run_phase6_transfer.py \
  --ckpt-dir results/wildfire_phase3_multiseed --out /tmp/transfer
```

Add `--quick` to any of the three for a tiny end-to-end validation run before committing to a
full sweep.

### Determinism and seed protocol

Determinism is enforced end to end: training reset seeds are `seed·100000 + episode` and eval
reset seeds are `group + 100000 + episode`, so the two streams are disjoint by construction and
no evaluation ignition is ever seen during training. The seeding pipeline is proven
bit-identical on CPU.

One documented exception exists (Greedy-Risk only, confined to that policy's collision-prone
trajectories, no ranking or conclusion affected) — see the reproducibility note in
[`docs/RESULTS_FROZEN.md`](docs/RESULTS_FROZEN.md).

---

## 🎛️ Configuration

Scenario conditions are selected by a named **regime**, defined once in
[`src/wildfire_marl/env/regimes.py`](src/wildfire_marl/env/regimes.py):

| Regime | Weather (vs ERA5) | Step (sim-min) | Ignition | Firebreak | Purpose |
|---|---|:---:|---|:---:|---|
| `legacy` | extreme (unchanged) | 60 | uniform | none | reproduce the original, unsuppressible results |
| `default` | moderated | 30 | threat (upwind of assets) | 3×3 | **main benchmark** (identical to `medium`) |
| `easy` / `medium` / `hard` | increasing severity | 20 / 30 / 40 | threat | 3×3 | difficulty axis for the robustness study |

Environments are constructed through a single factory so every method shares one env
definition:

```python
from wildfire_marl.env.regimes import make_marl_env

env = make_marl_env("saudi", regime="default", num_agents=3)
```

Keyword overrides win over regime defaults. YAML training configs under `configs/marl/` select a
region and regime and set the learning budget; they do not redefine physics knobs. See
[`configs/marl/regimes/README.md`](configs/marl/regimes/README.md).

> [!NOTE]
> **Cell2Fire FBP spread physics is never modified** — only scenario weather inputs, ignition
> placement, and the wrapper's firebreak footprint change.

---

## 🗂️ Repository layout

```text
src/wildfire_marl/        Library
  env/                    Cell2Fire binding, Gymnasium + PettingZoo envs, regimes, rewards
  agents/                 information-matched heuristics, PPO baseline, comm/strategic networks
  train/                  MAPPO / QMIX / CommNet / HierComm training loops
  eval/                   metrics, significance testing, transfer, evaluation protocol
  data/                   region tensors -> Cell2Fire landscapes; frozen CHANNEL_ORDER
  infra/                  critical-infrastructure rasters and cascade model
  viz/                    rollout, comparison and transfer rendering
  reproducibility/        seeding, manifests, integrity guards

scripts/                  21 experiment drivers (run_phase3/4/6, freeze, render, build, verify)
configs/                  YAML experiment + regime configs
tests/                    pytest suite
notebooks/                per-source geospatial acquisition pipelines (ERA5/FIRMS/DEM/NDVI)
results/                  frozen, fingerprinted result artifacts — see results/README.md
dashboard/                React results dashboard, built from the frozen artifacts
data/                     Cell2Fire landscapes + committed sample tensor (rasters git-ignored)
docs/                     data card, infra card, frozen results, migration, review history
docs/media/               README rollout GIFs
third_party/firehose/     vendored Firehose fork of Cell2Fire (GPL-3.0)
```

Full-resolution rollout media (GIF/MP4) is written to `figures/` and is **not tracked** — it is
regenerated with `scripts/render_phase5_gifs.py` and `scripts/render_phase6_transfer.py`, and
the web-sized copies that actually ship live in `dashboard/public/media/`.

---

## 📈 Results dashboard

`dashboard/` is a static, bilingual (English / Arabic with full RTL), themed React site.
**Every number it displays is generated at build time from the frozen artifacts — none is
hand-typed.** The build re-hashes each frozen directory and aborts on any fingerprint
mismatch.

```bash
python scripts/build_dashboard_data.py --copy-media   # frozen results -> dashboard JSON
python scripts/check_dashboard_consistency.py         # paper <-> dashboard number gate
cd dashboard && npm ci && npm run dev
```

Add `--release` to the consistency check to additionally require every referenced media file
and replay to exist on disk — mandatory before deploying.

> [!TIP]
> 📖 **[`dashboard/README.md`](dashboard/README.md) is the complete dashboard guide.** It covers
> the full workflow (data regeneration, release gates, dev/test/build), what is committed versus
> generated, the documented design decisions, the WSL-only steps for simulation-backed artifacts
> (Tier-2 interactive replays and MP4 compression), the Arabic native-speaker review gate, and
> the release checklist. Read it before changing anything under `dashboard/`.

---

## 🔒 Reproducibility guarantees

Each directory under `results/` carries a `MANIFEST.sha256` (a hash per file) and a
`FREEZE.json` (one fingerprint over that manifest), so any change to a reported number is
detectable. Fingerprints are computed over paths relative to each directory, so artifacts
verify identically wherever the tree is checked out.

CI enforces, on every push:

- ✅ determinism of the seeding pipeline and validity of the committed sample tensor;
- ✅ seed integrity across checkpoints, manifests and registered evaluation CSVs;
- ✅ that the dashboard JSON still matches both the frozen summaries and the paper's values;
- ✅ accessibility (axe-core over every route × locale × theme) and Lighthouse budgets;
- ✅ lint (ruff), formatting (black), tests on Python 3.10 and 3.11, and a clean-install check;
- ✅ that no tracked file exceeds 50 MB.

See [`results/README.md`](results/README.md) for the per-directory fingerprint table.

---

## 🛠️ Development

### Local checks

These mirror what CI runs:

```bash
ruff check src tests scripts        # lint
black --check src tests scripts     # formatting
mypy                                # type check (config in pyproject.toml)
pytest -q                           # tests
```

Install the pre-commit hooks once; they then run ruff, black, `nbstripout`, and the large-file
guard on every commit:

```bash
pre-commit install
```

### Integrity gates

```bash
python -m wildfire_marl.reproducibility.validate_tensors      # committed tensor validity
python -m wildfire_marl.reproducibility.check_seed_integrity  # seed/manifest integrity
python scripts/check_dashboard_consistency.py                 # paper <-> dashboard numbers
```

### Testing

```bash
pytest tests/                       # full suite
pytest -m "not slow"                # skip slow tests
pytest --cov=wildfire_marl          # with coverage
```

Markers are declared in `pyproject.toml`: `slow` and `integration` (end-to-end tests requiring
optional dependencies).

---

## 🩺 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | The `[dev]` extra excludes PyTorch | `pip install "torch>=2.1,<3.0"` |
| `sim_smoke.py: error: the following arguments are required: --region` | `--region` is mandatory | `python scripts/sim_smoke.py --region saudi` |
| 7 tests skipped | Cell2Fire binary not built — expected | Build the simulator, or ignore if verifying only |
| Tests skipped on Windows | The simulator is Linux-only | Use WSL2 / Linux / Colab for simulator work |
| Evaluation starts a long training run | Checkpoints missing; drivers train what they cannot find | Restore checkpoints, or restrict `--train-seeds` |
| Dashboard build aborts on fingerprint mismatch | A frozen artifact changed | Intended — investigate the change; do not bypass |
| `No dashboard data at ... — run scripts/build_dashboard_data.py` | Consistency check ran before the build | Run `build_dashboard_data.py` first |

---

## ⚠️ Limitations

Stated plainly, and documented at length in [`docs/data_card.md`](docs/data_card.md):

- **Research use only.** Not intended for operational firefighting decisions.
- **Single month, two regions.** Temporal coverage is June 2025 — a documented seasonality
  limitation.
- **Simplified physics.** No combustion chemistry, ember transport, or atmospheric coupling.
- **Stylized asset values.** Criticality is a Gaussian falloff around public infrastructure
  sites, not measured economic valuations (see [Metrics](#metrics)).
- **Declared fuel-mapping sensitivity.** NDVI → FBP fuel thresholds are expert judgment and a
  declared sensitivity parameter; California NDVI is reconstructed under a documented
  assumption (a recorded data debt).
- **Per-region normalization.** Min-max is per region, so cross-region transfer requires a
  shared scaler.
- **Three-agent scale.** The communication null result is established at `N = 3`.

---

## 📚 Provenance

| | |
|---|---|
| 🧊 Frozen results, protocol, caveats | [`docs/RESULTS_FROZEN.md`](docs/RESULTS_FROZEN.md) |
| 🗺️ Data sources, licenses, CRS, normalization | [`docs/data_card.md`](docs/data_card.md) |
| 🏭 Critical-infrastructure layer | [`docs/infra_card.md`](docs/infra_card.md) |
| 🔁 V1 → V2 re-founding on Cell2Fire | [`docs/MIGRATION.md`](docs/MIGRATION.md) |
| 🔍 Independent audit + review history | [`docs/reviews/`](docs/reviews/) |
| 📦 Claim → command → expected number | [`docs/supplement_assets/REPRODUCIBILITY.md`](docs/supplement_assets/REPRODUCIBILITY.md) |
| ⚗️ Vendored simulator provenance | [`third_party/README.md`](third_party/README.md) |
| 📜 Release history | [`CHANGELOG.md`](CHANGELOG.md) |

---

## 🤝 Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow and
standards, and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community expectations.

Core standards enforced here: one source of truth per concern (one env, one metrics module),
no hardcoded paths or magic numbers, determinism via the environment's seeded RNG rather than
global `np.random`, and no large binaries in git.

---

## 📝 Citation

If you use this software, please cite it:

```bibtex
@software{akarma_wildfire_marl,
  author  = {Akarma, Ali},
  title   = {wildfire-marl: Hierarchical Multi-Agent Reinforcement Learning
             for Wildfire Suppression on Validated Fire Physics},
  url     = {https://github.com/aliakarma/wildfire-rl},
  version = {2.0.0a0}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable record.

### Citing the underlying simulators

This work builds directly on two prior systems and they should be cited alongside it:

- **Cell2Fire** (Pais et al.) — the peer-reviewed cell-based fire growth simulator implementing
  the Canadian FBP system.
- **Firehose** (Shen, Curtis et al., 2022) — the Cell2Fire fork providing the interactive
  stdin/stdout step protocol this environment builds on.

Provenance, the fork's exact commit, and the physics-integrity diff against upstream Cell2Fire
are recorded in [`third_party/README.md`](third_party/README.md) and
[`third_party/firehose/VENDORED.md`](third_party/firehose/VENDORED.md).

---

## ⚖️ License

MIT for this repository's code — see [`LICENSE`](LICENSE).

The vendored simulator under `third_party/firehose/` is the **Firehose** fork of Cell2Fire,
licensed **GPL-3.0**. It retains its own license and is redistributed unmodified relative to its
upstream commit; its fire physics is byte-identical to Cell2Fire, with changes confined to
interactive loop control and debug-logging suppression.
