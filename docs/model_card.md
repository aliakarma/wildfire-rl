# Model Card — Wildfire-RL PPO

## Model details

- **Architecture:** PPO (Stable-Baselines3) with a custom CNN feature extractor
  (`wildfire_rl.models.cnn.CustomCNN`): Conv(8→32)–ReLU–[MaxPool]–Conv(32→64)–ReLU–
  [MaxPool]–Flatten–Linear(→256)–ReLU. The first conv's `in_channels` is read from the
  observation space, so it tracks the channel count automatically.
- **Action space:** `Discrete(5)` (up/down/left/right/stay) single-agent;
  `MultiDiscrete([5]*N)` for centralized MARL.
- **Observation:** `(8, G, G)` float32 in [0, 1] — the 7-channel state tensor plus one
  agent-position/occupancy channel (Markov fix, `EnvConfig.include_agent_channel`). Checkpoints
  trained on the earlier 7-channel observation live in `models/deprecated_pre_markov/` and are
  incompatible with the current environment.
- **Default hyperparameters:** `lr=3e-4`, `n_steps=2048`, `batch_size=256`, `gamma=0.99`,
  `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`, `total_timesteps=100k`
  (see `configs/ppo/default.yaml`). Identical across regions for clean domain-shift comparison.
- **Checkpoint size:** with `cnn_pooling=true` the flattened dim shrinks ~16×, producing far
  smaller checkpoints than the original no-pooling network (which yielded ~200 MB files).

## Training data

Region state tensors (see [data_card.md](data_card.md)). Models are trained per region and
per seed (`seeds=[0..4]`).

## Evaluation

- Metrics: episode reward, burned cells (threshold 0.5), fire intensity, containment_rate — over
  50 episodes / 5 disjoint seeds, with bootstrap 95% CIs and Cohen's d.
- **Always reported against baselines** (`RandomPolicy`, `NoOpPolicy`) **and the heuristic routers**
  (`NearestFirePolicy`, `FrontierPolicy`). The heuristics are the **effective method**; single-agent
  PPO does **not** beat no-op on this task (a rigorous **negative result**, enforced by the
  effective-method gate). The heuristics are privileged (oracle localization); PPO is not — this
  asymmetry is stated wherever a heuristic outperforms PPO.
- Cross-region transfer is reported as the full symmetric matrix; compare a transferred
  policy to the **native** policy on the same target environment (not across environments).

## Intended use & limitations

- Research artifact for studying RL wildfire suppression and ecological domain transfer.
- **Not** for operational fire management. The simulator omits real fire physics; absolute
  rewards depend on fuel load and are not physically calibrated.
- A single firefighting agent covers a small patch per step; results should always be read
  relative to the random/no-op baselines.

## Distribution

Checkpoints are hosted on the Hugging Face Hub (not git). Fetch + verify:

```bash
python scripts/fetch_models.py --repo aliakarma/wildfire-rl-ppo --verify results/models_manifest.json
```

## Reproducibility

Each model ships a `results/runs/train_<region>_seed_<s>.json` with git SHA, config hash,
seed, and library versions. See [reproducibility.md](reproducibility.md).
