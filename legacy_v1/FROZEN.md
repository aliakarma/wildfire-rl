# legacy_v1 — FROZEN (read-only archive)

**Status:** Frozen as of 2026-07-04 (Phase 0 of `REMEDIATION_PLAN_V2.md`, branch `v2-cell2fire`).

This directory is the complete, unmodified V1 prototype: the hand-rolled wildfire suppression
simulator, its oracle heuristics, the stubbed hierarchical controller, and every script, config,
test, result CSV, and figure produced with them. It is preserved as a **citable historical
artifact** — the honest-negative-result prototype that motivated the V2 re-founding — and is
**not part of the V2 contribution**.

## Rules

1. **Nothing in `legacy_v1/` is ever cited as a result.** It is prior art / motivation only.
2. **No file here is edited, extended, or imported by V2 code.** The reusable evaluation and
   reproducibility core was *ported* (copied, with import paths updated) into
   `src/wildfire_marl/` in Phase 0; see `docs/MIGRATION.md` for the exact mapping.
3. V1 result CSVs under `legacy_v1/results/` remain byte-identical to their committed state so
   the V1 provenance chain (manifests, hashes, `check_seed_integrity`) stays auditable.

## Why V1 was retired

Three reasons an AAAI main-track reviewer rejects on sight (see `docs/MIGRATION.md`):

1. **Toy simulator** — `src/wildfire_rl/envs/dynamics.py` is a hand-rolled cellular fire model,
   not validated physics. Replaced by Cell2Fire (Phase 1).
2. **Oracle heuristics** — baselines read privileged env state (`env.agent_pos`), confounding
   every comparison. Replaced by information-matched policies (Phase 4).
3. **Stubbed learned controller** — the "hierarchical RL" high level was hand-coded, not
   learned. Replaced by a genuinely trained hierarchical MARL policy (Phases 5–7).

## Contents

| Path | What it is |
|------|------------|
| `src/wildfire_rl/` | The full V1 package (envs, coordination, routing, ablation, eval, viz, train) |
| `scripts/` | V1 pipeline scripts (train, evaluate, ablations, manifests, figures) |
| `tests/` | V1 test suite (env/MARL/ablation tests; ported-module tests now live in `/tests`) |
| `configs/` | V1 OmegaConf configs |
| `results/`, `figures/` | V1 result CSVs, manifests, and renders (never cited as V2 results) |
| `notebooks/`, `experimental/`, `Colab_*.ipynb` | V1 notebooks and exploratory work |
| `models/` | V1 checkpoints (git-ignored, ~5.7 GB on local disk only) |
| `docs/` | V1 documentation (architecture, model card, reproducibility, V1 paper report) |
| `Makefile`, `Dockerfile`, `requirements*.txt`, `environment.yml` | V1 build/env surface |
