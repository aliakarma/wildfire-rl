# MIGRATION.md — V1 → V2 (Cell2Fire re-founding)

**Date:** 2026-07-04 · **Branch:** `v2-cell2fire` · **Phase:** 0 of `REMEDIATION_PLAN_V2.md`

This document records what carried over from the V1 prototype, what was retired, and why.
The V1 surface is frozen, unmodified, under [`legacy_v1/`](../legacy_v1/FROZEN.md). The V2
package is `src/wildfire_marl/`. **Nothing from `legacy_v1/` is ever cited as a result** — it
is prior art / motivation only, and no V2 code imports from it.

## Why the re-founding: the three AAAI rejection reasons this migration fixes

1. **Hand-rolled toy simulator.** V1's fire model (`legacy_v1/src/wildfire_rl/envs/dynamics.py`)
   is an invented cellular automaton with hand-tuned constants (`base_spread=0.01`,
   `fuel_coeff=0.15`, …). No amount of statistical rigor on top of unvalidated physics makes the
   conclusions about *wildfire* credible. **Fix:** Phase 1 adopts Cell2Fire, a peer-reviewed,
   physics-validated fire-growth simulator (via a fresh Gymnasium binding modeled on the
   Firehose RL wrapper). Cell2Fire physics is never modified.

2. **Oracle-privileged heuristics.** V1 baselines read privileged environment state
   (`env.agent_pos`, true fire fields) that the learned policies could not see, confounding
   every comparison in both directions. **Fix:** Phase 4 rebuilds all baselines
   information-matched — every policy reads only the observation channels.

3. **Stubbed "learned" controller.** V1's hierarchical high level was a hand-coded rule set
   (`legacy_v1/src/wildfire_rl/coordination/strategic_controller.py`), not a learned policy, so
   the paper had no genuine ML contribution. **Fix:** Phases 5–7 train a real decentralized
   cooperating MARL team (MAPPO/QMIX, PettingZoo) plus a *learned* hierarchical strategic layer,
   gated by the Phase-8 learned-beats-heuristics test.

A fourth, related reason — **invented infrastructure parameters** — is fixed in Phase 3 (real
GIS asset data with provenance + sensitivity analysis).

## What carried over (ported in Phase 0, logic unchanged)

| V1 (frozen at) | V2 (ported to) | Notes |
|---|---|---|
| `legacy_v1/src/wildfire_rl/eval/metrics.py` | `src/wildfire_marl/eval/metrics.py` | Verbatim. ISR/WEL/CPS/RAC/PCA/PA/CCL/CE + core metrics; re-anchored to Cell2Fire final state in Phase 3. |
| `legacy_v1/src/wildfire_rl/eval/significance.py` | `src/wildfire_marl/eval/significance.py` | Verbatim (docstring cross-reference updated) plus one compatibility guard: SciPy >= 1.14 no longer raises on all-zero Wilcoxon differences, so `wilcoxon_test` now checks that case explicitly and returns NaN (the documented V1 contract; its ported test enforces it). |
| `legacy_v1/src/wildfire_rl/eval/transfer.py` | `src/wildfire_marl/eval/transfer.py` | TRS/CDGG/adaptation-asymmetry verbatim. **One change:** `transfer_matrix` takes `evaluate_fn` as an argument instead of importing the retired V1 env-coupled evaluation loop (re-anchored in Phase 9). |
| `legacy_v1/src/wildfire_rl/seeding.py` | `src/wildfire_marl/reproducibility/seeding.py` | Verbatim (docstring updated). torch/SB3 remain lazy imports. |
| `legacy_v1/src/wildfire_rl/logging_utils.py` | `src/wildfire_marl/reproducibility/logging_utils.py` | Verbatim except default logger name (`wildfire_marl`) and `pettingzoo` added to recorded library versions. |
| `legacy_v1/src/wildfire_rl/data/manifest.py` | `src/wildfire_marl/reproducibility/manifest.py` | Verbatim. sha256 build/write/verify manifests. |
| `legacy_v1/src/wildfire_rl/paths.py` | `src/wildfire_marl/paths.py` | Verbatim. `WILDFIRE_*` env-var overrides preserved. |
| `legacy_v1/scripts/check_seed_integrity.py` | `src/wildfire_marl/reproducibility/check_seed_integrity.py` | Logic unchanged; now a `python -m` module. V2 eval CSVs are registered in `EVAL_CSVS` as phases produce them (V1 CSVs are never re-certified). Learned-policy prefix set widened to `ppo|mappo|qmix`. |
| `legacy_v1/scripts/make_manifest.py` | `src/wildfire_marl/reproducibility/make_manifest.py` | Logic unchanged; `_bootstrap` shim dropped (proper package install). |
| `legacy_v1/scripts/validate_tensors.py` | `src/wildfire_marl/reproducibility/validate_tensors.py` | Logic unchanged; channel contract now from `wildfire_marl.data.CHANNEL_ORDER`; data root from `wildfire_marl.paths`. |
| `legacy_v1/scripts/build_report_tables.py` | `src/wildfire_marl/reproducibility/build_report_tables.py` | Rendering mechanism unchanged; `TABLES` starts empty and is re-registered per phase (V1 tables reference retired results). |
| `legacy_v1/tests/{test_metrics,test_significance,test_seeding,test_logging}.py` | `tests/` | Imports re-pointed at `wildfire_marl`. The torch determinism test uses `pytest.importorskip` because torch is now an optional extra. |
| `data/<region>/` rasters + tensors | kept in place | NDVI/DEM/ERA5/FIRMS-derived 7-channel tensors feed the Phase-2 Cell2Fire landscape conversion. `CHANNEL_ORDER` frozen in `wildfire_marl.data`. |
| `docs/data_card.md` | kept in place | Provenance note added; V1 processing references now point into `legacy_v1/`. |

Also carried over as *design* (re-implemented on the new stack in later phases, not code-ported
in Phase 0): the symmetric transfer methodology (`scenario_seed_offset`, disjoint eval seeds),
the strategic-metric design, the hierarchical-controller idea, the coordination-ablation design,
and the honest negative-result framing with the effective-method gate.

## What was retired (frozen in `legacy_v1/`, never imported)

| Retired | Why |
|---|---|
| `envs/dynamics.py`, `envs/base.py`, `envs/multi_agent*.py`, `envs/hybrid_multi_agent.py`, scenario generators | The toy simulator (rejection reason 1). Replaced by Cell2Fire + fresh Gymnasium/PettingZoo wrappers (Phases 1, 5). |
| `routing/`, `coordination/` (incl. `strategic_controller.py`) | Oracle-privileged heuristics (rejection reason 2) and the hand-coded strategic stub (rejection reason 3). Replaced by information-matched heuristics (Phase 4) and a learned strategic layer (Phase 7). |
| `ablation/`, V1 `scripts/`, V1 `configs/` | Bound to the retired env/controllers. Rebuilt against the new stack in Phases 4–10. |
| V1 `results/`, `figures/`, notebooks | Historical artifacts of the toy-sim experiments. Never cited as V2 results. |
| `cli.py`, `train/ppo.py`, `models/cnn.py`, SB3-centric packaging | V2 is MARL-first (`gymnasium` + `pettingzoo`); SB3 returns only as an optional Phase-4 baseline extra. |
| Invented infrastructure parameters (`InfraConfig` defaults) | Rejection reason for the domain claim. Replaced by real, licensed GIS asset data in Phase 3. |

## Platform note

V1 was developed on native Windows. From Phase 0 onward all development happens on **Linux**
(WSL2 Ubuntu here; any Linux/Colab/cluster works) because Cell2Fire is a C++ build (Eigen +
`make`) and Firehose targets old `gym`. `environment-linux.yml` pins the Linux toolchain;
the Phase-0 validation (ported tests + gcc/cmake/make/Eigen check) runs under WSL2.
