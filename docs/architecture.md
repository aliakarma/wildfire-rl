# Architecture

## Package layout (`src/wildfire_rl/`)

| Module | Responsibility | Replaces (notebook origin) |
|---|---|---|
| `config.py` | OmegaConf dataclass schema + loader | scattered constants in every notebook |
| `paths.py` | Repo-relative, env-overridable path resolution | `/content/drive/MyDrive/...` hardcoded paths |
| `seeding.py` | Global + per-generator RNG seeding | the single `set_seed` in `07b` |
| `logging_utils.py` | Logger + run-metadata (git SHA, config hash, versions) | bare `print`s |
| `data/normalize.py` | One min-max + shared-scaler helpers | `normalize()` copied into 01/03/04 |
| `data/tensor_stack.py` | Channel stacking → `state_tensor.npy` (+metadata) | `05_california_tensor_stacking` (+ missing Saudi) |
| `data/manifest.py` | sha256 integrity manifests | — (new) |
| `envs/dynamics.py` | Pure spread / suppression / decay / ignition | inlined in 6 env copies |
| `envs/base.py` | `WildfireEnv` (single agent) | `SaudiWildfireEnv`, `CaliforniaWildfireEnv`, `SaudiEvaluationEnv`, `CaliforniaEvaluationEnv` |
| `envs/multi_agent.py` | `MultiAgentWildfireEnv` (centralized) | two `MultiAgentSaudiEnv` copies |
| `models/cnn.py` | `CustomCNN` feature extractor | ~7 copies |
| `train/ppo.py` | PPO build + train | Saudi/California training loops |
| `eval/metrics.py` | `burned_cells`, `fire_intensity`, `summarize` | inline metric code (inconsistent thresholds) |
| `eval/baselines.py` | `RandomPolicy`, `NoOpPolicy` | **missing** in original "baseline" notebook |
| `eval/evaluate.py` | One evaluation loop (seeded per episode) | duplicated `evaluate_model` |
| `eval/transfer.py` | Full N×N transfer matrix | asymmetric, partly-hardcoded transfer CSVs |
| `experiments/transfer_run.py` | Orchestrate transfer from config + models | `07` + `10` |
| `viz/figures.py` | Deterministic figure generation | ad-hoc plotting cells |
| `cli.py` | `wildfire-rl` entrypoint | notebook execution |

## State tensor

Shape `(7, H, W)`, channels frozen in this order:

| idx | channel | source | meaning |
|---|---|---|---|
| 0 | fire | FIRMS active fire | ignition / fire intensity (0–1) |
| 1 | fuel | MODIS NDVI | vegetation/fuel density (0–1) |
| 2 | wind_x | ERA5 10m u | zonal wind |
| 3 | wind_y | ERA5 10m v | meridional wind |
| 4 | terrain | SRTM DEM | slope (min-max) |
| 5 | temperature | ERA5 2m T | temperature |
| 6 | humidity | ERA5 2m dewpoint | humidity proxy |

## Environment dynamics (per `step`)

1. **Move** the agent(s) (5 discrete actions: up/down/left/right/stay).
2. **Suppress**: fire in the `(2r+1)²` patch around each agent is multiplied by
   `suppression_factor`.
3. **Spread** (vectorized, seeded): a cell with ≥1 burning orthogonal neighbor ignites with
   `p = base + fuel·f + wind·w + terrain·t`, drawing from `self.np_random`.
4. **Decay + deplete**: fire `×= decay`, sub-threshold fire zeroed, fuel reduced.
5. **Reward**: `-Σ fire (+ bonus)`; `reward_mode="normalized"` divides by initial fire for
   cross-region comparability.
6. **Terminate** when total fire `< termination_fire_threshold`; **truncate** at `max_steps`.

All constants are `EnvConfig` fields — see `configs/env/single.yaml`.

## Data flow

```
download_data.py ──▶ data/<region>/raw/{era5,firms,dem,ndvi}
build_tensors.py ──▶ data/<region>/grids/GxG/{<channel>.npy, state_tensor.npy}
train.py        ──▶ models/ppo_<region>_<G>_seed_<s>.zip + results/runs/*.json
evaluate.py     ──▶ results/eval_<region>.json
transfer.py     ──▶ results/transfer_matrix.csv
make_figures.py ──▶ figures/*.png
```

## Design principles

- **One implementation per concept** (env, CNN, metrics, training, eval).
- **Config over code**; **seed everything**; **artifacts are regenerable or fetchable**.
- **Notebooks orchestrate, `src/` computes.**
