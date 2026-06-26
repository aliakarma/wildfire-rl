# Reproducibility

This project targets artifact-evaluation-grade reproducibility. This document is the
protocol a reviewer should follow, and the honest list of current limitations.

## One-command reproduction

```bash
conda env create -f environment.yml && conda activate wildfire-rl && pip install -e .
make reproduce        # test → train → evaluate → transfer → figures
```

For a fast sanity pass (CI-sized):

```bash
wildfire-rl train --set ppo.total_timesteps=2000 seeds=[0]
wildfire-rl evaluate --set region.name=saudi region.dir=saudi_eastern_province
```

## Determinism

- `wildfire_rl.seeding.set_global_seed(seed)` seeds Python, NumPy, PyTorch, and SB3, and
  sets `PYTHONHASHSEED` + deterministic cuDNN.
- The environment dynamics draw from `self.np_random` (seeded by `reset(seed=...)`), **not**
  the global `np.random`. Same seed + same actions ⇒ identical trajectories
  (`tests/test_env_api.py::test_determinism_same_seed`).
- Evaluation seeds each episode as `reset(seed=base_seed + episode_index)`.

## Run metadata

Every training/transfer run writes `results/runs/*.json` with: UTC timestamp, git SHA,
config hash, seed, and resolved library versions. This ties each artifact to the exact code
and configuration that produced it.

## Data & model integrity

```bash
make manifest         # results/data_manifest.json, results/models_manifest.json (sha256+bytes)
python scripts/fetch_models.py --repo <hf-repo> --verify results/models_manifest.json
```

## Environment locking

- `requirements.txt` pins runtime deps; `environment.yml` pins the full conda stack
  (including GDAL/rasterio). Regenerate exact hashes with `pip-compile` if needed.
- `Dockerfile` provides a pinned CPU runtime for clean-room reproduction.

## Known limitations (be explicit with reviewers)

1. **Scenario diversity.** By default the env starts from one fixed ignition tensor per
   region (original behavior). Set `env.randomize_ignition=true` to sample a *distribution*
   of held-out scenarios — required to measure generalization rather than memorization.
2. **Cross-region normalization.** Channels are min-max normalized per region. For
   transfer claims, fit a shared scaler (`data/normalize.fit_shared_minmax`) so absolute
   magnitudes are comparable.
3. **Reward scale.** `reward_mode="raw"` is not comparable across regions with different
   fuel loads; use `reward_mode="normalized"` for cross-region/cross-team comparisons.
4. **Spread RNG semantics.** The refactor uses one Bernoulli draw per target cell
   (vectorized, seeded) rather than the original per-(source,neighbor) draw; expected
   behavior is preserved but exact legacy numbers are not bit-reproducible.
5. **Eval N.** Default 20 episodes / 5 seeds; increase for tighter confidence intervals.
