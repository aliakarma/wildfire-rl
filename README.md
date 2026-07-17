# wildfire-marl — Multi-Agent Wildfire Suppression on Validated Fire Physics

> **Project status:** this repository is founded on a peer-reviewed, physics-validated fire
> simulator ([Cell2Fire](https://github.com/cell2fire/Cell2Fire), via a fresh Gymnasium
> binding modeled on the Firehose RL wrapper). An earlier hand-rolled prototype simulator
> (V1) has been removed; nothing from it is cited as a result.

## The research bet

*A learned, coordinated, infrastructure-aware multi-agent RL policy beats strong
information-matched heuristics and naïve deep RL on validated fire physics.* Firehose
independently reports that plain PPO fails to train on larger random-ignition suppression;
beating that failure with a coordinated hierarchical method is a real — but not guaranteed —
contribution. Phase 8 of the plan is a hard go/no-go gate: if learning does not beat
heuristics, the documented fallback is an honest benchmark/negative-result paper.

## What this project is

- **Environment:** Cell2Fire (C++, validated fire growth) wrapped for RL — single-agent
  Gymnasium first (Phase 1), then a PettingZoo `ParallelEnv` with N firefighting agents under
  partial observability (Phase 5). Cell2Fire physics is never modified; all suppression/agent
  logic lives in the wrapper.
- **Data:** real 7-channel geospatial tensors (NDVI, DEM, ERA5 weather, FIRMS hotspots) for
  Saudi Arabia's Eastern Province (desert-petroleum regime) and Northern California
  (forest-WUI regime), converted into Cell2Fire landscapes with one shared fuel model
  (Phase 2). See [`docs/data_card.md`](docs/data_card.md).
- **Infrastructure:** real, provenance-documented critical-asset data (Saudi petroleum
  facilities; California critical facilities/WUI) as a criticality observation channel and a
  value-weighted reward (Phase 3).
- **Method:** decentralized cooperating MARL (MAPPO/QMIX, CTDE, shared team reward) plus a
  learned hierarchical strategic layer that prioritizes which assets/sectors the team defends
  (Phases 5–7).
- **Evaluation:** information-matched baselines, bootstrap CIs + effect sizes, disjoint
  train/eval ignition seeds, symmetric Saudi↔California transfer (TRS/CDGG/asymmetry),
  single-factor ablations, and full artifact provenance (Phases 4, 8–12).

## Repository layout (V2)

```
src/wildfire_marl/          V2 package
  env/                      Cell2Fire binding + Gymnasium/PettingZoo envs (Phase 1+)
  agents/                   information-matched heuristics, PPO baseline, MARL, strategic layer
  data/                     region tensors -> Cell2Fire landscapes (Phase 2); frozen CHANNEL_ORDER
  infra/                    real-data critical-infrastructure layer (Phase 3)
  eval/                     metrics / significance / transfer   [ported from V1, tests green]
  reproducibility/          seeding, manifests, integrity guards [ported from V1, tests green]
  viz/, experiments/        visualization + experiment drivers (later phases)
tests/                      ported test suite (runs on Linux)
data/                       preserved region rasters + tensors (feed Phase 2)
docs/                       data card, MIGRATION.md, RESULTS_FROZEN.md
```

## Getting started (Linux only)

Development happens on Linux (WSL2 / native / Colab) — Cell2Fire is a C++ build and the
native-Windows path is explicitly abandoned (see the plan's platform requirements).

```bash
# toolchain + env
conda env create -f environment-linux.yml
conda activate wildfire-marl

# or plain pip in any Linux Python >= 3.10
pip install -e ".[dev]"

# verify the ported core
python -c "import wildfire_marl; from wildfire_marl.eval import metrics, significance, transfer; print('core ported')"
pytest tests/
```

## Provenance

- What carried over from V1, what was retired, and the three AAAI rejection reasons the
  migration fixes: [`docs/MIGRATION.md`](docs/MIGRATION.md)
- Peer-review remediation plan and its execution log:
  [`PEER_REVIEW_REMEDIATION_PLAN.md`](PEER_REVIEW_REMEDIATION_PLAN.md), [`peer_phases.md`](peer_phases.md)
- Frozen canonical main results for the paper: [`docs/RESULTS_FROZEN.md`](docs/RESULTS_FROZEN.md)

## License

MIT for this repository's code. Cell2Fire (added as a submodule in Phase 1) is GPL-3.0;
licensing implications for the release are documented in the plan (Phase 15).
