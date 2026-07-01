# models/

Trained PPO checkpoints live here at runtime but are **not tracked in git** (each is
~13 MB; the current root set is ~1.2 GB). They are distributed via the Hugging Face Hub
and verified against `results/models_manifest.json`.

```bash
python scripts/fetch_models.py --repo aliakarma/wildfire-rl-ppo \
    --verify results/models_manifest.json
```

## Current checkpoints (models/ root)

The latest research phase: hybrid coordination strategies and the adaptive (v6) policy.

- `ppo_hybrid_marl_<region>_<N>agents_<strategy>_seed_<seed>.zip` — strategy is
  `nearest_fire` or `frontier` (produced by `scripts/train_single_hybrid_marl.py`).
- `ppo_adaptive_marl_<region>_<N>agents_seed_<seed>.zip` — v6 adaptive coordination
  (produced by `scripts/train_adaptive_v6.py`).

## `Legacy/` — superseded checkpoints, kept for reproducibility

Older research phases are archived under `models/Legacy/`, not deleted, because the
evaluation scripts that regenerate the committed `results/v2`–`v3` and `figures/v2`–`v3`
tables still load them by name:

- `Legacy/v1_marl/` — base centralized MARL, `ppo_marl_<region>_<N>agents_seed_<seed>.zip`
  (`scripts/train_single_marl.py`, evaluated by `scripts/run_marl_evaluation.py`).
- `Legacy/v2_marl/` — entropy-coefficient sweep,
  `ppo_v2_marl_<region>_<N>agents_ent<coef>_seed_<seed>.zip`
  (`scripts/train_single_marl_v2.py` / `run_marl_v2_evaluation.py`).
- `Legacy/v3_marl/` — scaling follow-up, `ppo_v3_marl_<region>_<N>agents_seed_<seed>.zip`
  (`scripts/train_single_marl_v3.py` / `run_marl_v3_evaluation.py`).
- Flat files directly under `Legacy/` (`ppo_california_32*`, `ppo_saudi_32*`, an older
  `ppo_marl_saudi_*` generation) — pre-MARL single-agent checkpoints, superseded.

`run_marl_v3_evaluation.py` loads v1 + v2 + v3 checkpoints together from a single
`models_dir()`, so to regenerate `results/v2` or `results/v3` without retraining, copy
the needed `Legacy/v1_marl` / `v2_marl` / `v3_marl` files back into `models/` root (or
set `WILDFIRE_MODELS_DIR` to a scratch folder containing all of them).

Everything in this directory except this README is git-ignored — see `.gitignore`.
