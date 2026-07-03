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

## Observation (Markov)

The **stored** `state_tensor.npy` is 7-channel (above). The **observation returned to the
policy** appends one agent channel, giving `(8, H, W)`:

| idx | channel | meaning |
|---|---|---|
| 0–6 | state tensor | fire, fuel, wind_x, wind_y, terrain, temperature, humidity |
| 7 | agent position | single-agent: one-hot agent cell; MARL: agent-occupancy map |

The agent channel makes the environment Markov for a movement policy
(`EnvConfig.include_agent_channel`, default `True`) — without it the raw state does not encode
where the agent is, and the policy cannot learn to navigate to the fire. Metrics
(`burned_cells`, `fire_intensity`) are computed from `env.state` (channel 0) and are unaffected
by the added channel. Checkpoints trained before this change (`models/deprecated_pre_markov/`)
expect 7-channel input and are incompatible.

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

## Critical infrastructure (Phase 15B.1)

The Saudi region models petroleum assets (`EnvConfig.infra`, `InfraConfig`):

- **Rasters** (`build_infrastructure.py` → `data/<region>/grids/32x32/infrastructure/`): `asset_type`
  (0=none, 1=refinery, 2=pipeline, 3=storage, 4=industrial), `criticality` ∈ [0,1], `blast_radius`.
- **Catastrophe penalty**: reward loses `catastrophe_weight · Σ value[type]·fire` — high-value assets
  are defended preferentially, not just total burned area.
- **Cascading detonation**: a burning asset ignites cells within its blast radius with `cascade_prob`
  (applied after spread; `info["cascade_ignited"]`).
- **Observation channel** (`observe_infra`): a normalized criticality channel stacked after the agent
  channel → obs is `(C+2, H, W)`. Off by default (the Phase-9 authoritative models are 8-channel).

## Hierarchical / hybrid control (Phase 15B.2)

The honest negative result (PPO fails at low-level navigation; heuristics succeed) motivates a
two-level controller that isolates the *learnable strategic* problem from the *un-learnable-here
navigation* problem:

```
 HIGH-LEVEL (strategic; greedy_risk | risk_aware | rl)
   obs: per-sector fire load + infra risk + agent positions
   act: rank sectors by risk, assign each agent to a sector
                         │  target sector per agent
 LOW-LEVEL (robust heuristic router: nearest_fire | frontier)
   compute_step_action → up/down/left/right/stay + deterministic suppression
```

- **`coordination/strategic_controller.py`** — `StrategicController` is an env-agnostic `Policy`
  (drop-in for `evaluate_policy`). Variants share one contract so they are directly comparable:
  `greedy_risk` (argmax fire load), `risk_aware` (fire + `infra_risk_weight`·fire-on-criticality →
  defends assets), `rl` (injected learned sector scorer; falls back to greedy).
- **`HierarchyConfig`** — `high_level`, `low_level`, `num_sectors`, `infra_risk_weight`.
- **Scientific point**: changing the high-level variant measurably changes infrastructure survival
  (ISR), while the low-level router stays stable — no collapse (`tests/test_strategic.py`).

Strategic outcomes are measured by the Phase 15B.3 metrics (`eval/metrics.py`): **ISR** (survival
rate), **WEL** (weighted economic loss), **CPS** (catastrophe prevention), **RAC** (risk-adjusted
containment), **PCA** (protected assets); and transfer by **TRS**/**CDGG** (`eval/transfer.py`).

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
