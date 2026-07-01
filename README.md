# 🔥 Wildfire-RL

**Geospatial reinforcement learning for wildfire suppression and cross-regional policy transfer (Saudi Arabia ↔ California).**

[![CI](https://github.com/aliakarma/wildfire-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/aliakarma/wildfire-rl/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Built with Gymnasium](https://img.shields.io/badge/RL-Gymnasium%20%2B%20SB3-brightgreen.svg)](https://gymnasium.farama.org/)

> A reproducible benchmark and framework for training PPO agents to contain wildfires on
> **real geospatial state tensors** (climate, terrain, vegetation, ignition), and for
> studying how suppression policies **transfer across ecological regimes** — desert
> (Saudi Eastern Province) vs. forest/mountain (Northern California).

---

## Overview

Wildfire-RL fuses remote-sensing data into a 7-channel `(fire, fuel, wind_x, wind_y, terrain, temperature, humidity)` state tensor and exposes it as a [Gymnasium](https://gymnasium.farama.org/) environment. A PPO agent (Stable-Baselines3) learns to move and suppress fire under stochastic, fuel/terrain/wind-driven spread dynamics. The framework supports:

- **Single-agent** wildfire suppression (PPO + custom CNN feature extractor).
- **Centralized cooperative multi-agent** control (MARL scaling: 1 / 3 / 5 agents).
- **Cross-regional transfer** evaluation (the full, symmetric Saudi ↔ California matrix).
- **Ablations** over dynamics components, and **multi-seed** statistical evaluation.
- **Baseline controls** (random / no-op) so learned-policy gains are measurable.

### Architecture

```
 Remote sensing            Preprocessing             RL                Evaluation
 ┌──────────┐   ┌───────────────────────────┐  ┌──────────────┐  ┌────────────────────┐
 │ ERA5     │   │ normalize → stack 7 channels│  │ WildfireEnv  │  │ baselines (rnd/noop)│
 │ FIRMS    │──▶│   data/<region>/grids/GxG/  │─▶│  + CustomCNN │─▶│ multi-seed metrics  │
 │ DEM/NDVI │   │      state_tensor.npy       │  │  PPO (SB3)   │  │ transfer matrix     │
 └──────────┘   └───────────────────────────┘  └──────────────┘  └────────────────────┘
```

> 📌 _Architecture/result figures live in `figures/` after `make figures` and in `docs/paper/`._

### Why cross-regional transfer?

Suppression policies are trained on a region's environmental tensor and evaluated **on the
same target environment as the native policy** — the only valid way to measure transfer.
This isolates *ecological domain shift* (sparse desert fuel vs. dense forest fuel, flat vs.
mountainous terrain) from raw fire-load differences. See [`docs/architecture.md`](docs/architecture.md)
and the methodological notes in [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Installation

```bash
# 1. Clone
git clone https://github.com/aliakarma/wildfire-rl.git
cd wildfire-rl

# 2. Create a pinned virtual environment (Python 3.10 or 3.11 — torch==2.3.1 has no 3.12+ wheels)
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash
# source venv/bin/activate          # Linux / macOS

# 3a. Exact, byte-reproducible install (recommended for reproducing results)
pip install -r requirements-dev.txt   # exact scientific + dev pins (torch==2.3.1, seaborn==0.13.2, ...)
pip install -e . --no-deps            # the wildfire_rl package itself
# After the first install, prefer the frozen transitive lock (see "Environment (pinned)"):
pip install -r requirements.lock.txt

# 3b. or conda (recommended if you need the geospatial preprocessing stack)
conda env create -f environment.yml
conda activate wildfire-rl
pip install -e . --no-deps

# Optional extras
pip install -e ".[geo]"     # rasterio/GDAL/EE/CDS — only to rebuild tensors from raw
pip install -e ".[track]"   # wandb + huggingface_hub
```

## Environment (pinned)

The authoritative, byte-reproducible environment for artifact evaluation:

```bash
python -m venv venv
source venv/Scripts/activate          # Windows Git Bash
pip install -r requirements-dev.txt   # exact scientific + dev pins
pip install -e . --no-deps            # editable package
pip install -r requirements.lock.txt  # full pinned transitive graph
```

- `requirements.txt` — exact runtime pins (incl. `scipy==1.13.0`, `seaborn==0.13.2`).
- `requirements-dev.txt` — the above plus exact test/lint pins.
- `requirements.lock.txt` — the complete frozen transitive graph (regenerate with
  `pip freeze --exclude-editable > requirements.lock.txt`). This file is the reference
  environment for reproducing every reported number.

> Python 3.10 or 3.11 is required: `torch==2.3.1` publishes no wheels for Python 3.12+.

**GPU (optional).** The pinned `torch==2.3.1` from PyPI is the **CPU** build (`2.3.1+cpu`), which is
the canonical, deterministic, CI-matched environment. For this workload (a small CNN on 32×32
grids) CPU is adequate. To train on an NVIDIA GPU, override with the CUDA wheel *after* the pinned
install (this changes the frozen graph; keep it out of `requirements.lock.txt`):

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Quickstart

```bash
wildfire-rl info                       # print version + resolved config
wildfire-rl evaluate --set region.dir=saudi_eastern_province region.name=saudi
                                       # runs random + no-op baselines (no GPU needed)
```

## Training

```bash
# Multi-seed PPO for a region (config-driven; no hardcoded paths)
wildfire-rl train --config configs/experiment/multiseed.yaml
# California:
wildfire-rl train --config configs/experiment/multiseed.yaml \
    --set region.name=california region.dir=california
# Quick CI-sized run:
wildfire-rl train --set ppo.total_timesteps=2000 seeds=[0]
```

## Evaluation & Transfer

```bash
# Evaluate a saved model against baselines
wildfire-rl evaluate --config configs/experiment/multiseed.yaml \
    --model models/ppo_saudi_32_seed_0.zip

# Full symmetric cross-region transfer matrix (incl. California→Saudi)
wildfire-rl transfer --config configs/experiment/transfer.yaml
```

## Reproducibility

```bash
make reproduce        # test → train → evaluate → transfer → figures
make manifest         # write sha256 manifests for data/ and models/
```
Every run writes `results/runs/*.json` capturing git SHA, config hash, library versions,
and seed. Determinism is enforced via `wildfire_rl.seeding.set_global_seed` **and** a
per-environment `np_random` generator. Full protocol: [`docs/reproducibility.md`](docs/reproducibility.md).

**PowerShell equivalents** (if `make` is unavailable):
```powershell
python -m pytest -q
python scripts/train.py --config configs/experiment/multiseed.yaml
python scripts/transfer.py --config configs/experiment/transfer.yaml
python scripts/make_figures.py
```

## Datasets

Raw remote-sensing data and derived tensors are **not** stored in git (licensing + size).
Rebuild them, or download a prepared bundle:

```bash
# Rebuild from raw (needs credentials in .env; see .env.example and [geo] extras)
python scripts/download_data.py --region saudi_eastern_province --source all
python scripts/build_tensors.py --region saudi_eastern_province --grid 32
```
A tiny synthetic sample lives in `data/sample/` for tests and the quickstart. Sources,
licenses, CRS, and temporal coverage are documented in [`docs/data_card.md`](docs/data_card.md).

## Model checkpoints

Trained PPO checkpoints (~200 MB each) are hosted on the Hugging Face Hub, not git:

```bash
python scripts/fetch_models.py --repo aliakarma/wildfire-rl-ppo \
    --verify results/models_manifest.json
```
See [`docs/model_card.md`](docs/model_card.md).

## Project structure

```
wildfire-rl/
├── src/wildfire_rl/      # installable package (envs, models, train, eval, viz, data)
├── configs/              # OmegaConf YAML (region / env / ppo / experiment)
├── scripts/              # certified CLI entrypoints (train, evaluate, transfer, ablation, marl)
├── tests/                # pytest suite (env API, determinism, metrics, config, CLI)
├── notebooks/            # demo / exploratory notebooks (outputs stripped)
├── docs/                 # architecture, reproducibility, data & model cards
├── data/   models/   results/   figures/   # artifacts (mostly git-ignored)
├── experimental/         # v2–v6 reward-shaping / coordination tracks (NON-certified)
├── reviews/history/      # archived third-party audits (provenance only)
└── .github/workflows/    # CI: lint, test, install, large-file guard
```

## Repository Layout Guarantee

The reproducible research pipeline consists of:
- `src/wildfire_rl/` — the installable package
- `scripts/{train,evaluate,transfer,run_ablation,run_marl_evaluation}.py`
- `configs/` and the canonical `results/*.csv`

Exploratory reward-shaping and coordination experiments (v2–v6) live under `experimental/`
and are **not** part of the certified reproducibility path. Archived third-party audits live
under `reviews/history/` for provenance only and must not be cited as project claims.

## Roadmap

- [ ] Decentralized MARL (MAPPO / QMIX) and inter-agent communication
- [ ] Held-out scenario distributions + partial observability
- [ ] Shared cross-region normalization for cleaner transfer attribution
- [ ] Higher resolution (64×64 / 128×128) with a vectorized spread kernel
- [ ] Additional regions (Australia, Mediterranean, Canada)
- [ ] Real-time forecasting from live ERA5/FIRMS feeds

## Citation

If you use this work, please cite it (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{akarma_wildfire_rl_2026,
  author  = {Akarma, Ali},
  title   = {Wildfire-RL: Geospatial Reinforcement Learning for Wildfire
             Suppression and Cross-Regional Policy Transfer},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/aliakarma/wildfire-rl}
}
```

## Acknowledgements

Inspired by the [PyroRL](https://github.com/) wildfire environment. Built on
[Gymnasium](https://gymnasium.farama.org/) and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Data: ECMWF ERA5,
NASA MODIS/SRTM/FIRMS (see the data card for licenses).

## License

Code is released under the [MIT License](LICENSE). Third-party datasets retain their own
licenses — see [`docs/data_card.md`](docs/data_card.md).
