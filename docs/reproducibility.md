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

## Observation (Markov)

- The policy **observation** appends an agent-position/occupancy channel to the 7-channel state
  tensor, so the network sees **8 channels** (`EnvConfig.include_agent_channel`, default `True`).
  Without it PPO cannot localize itself and collapses to a no-op — the original audit's root cause.
  Checkpoints trained on the earlier 7-channel observation are quarantined in
  `models/deprecated_pre_markov/` and are not loadable against the current CNN.

## Determinism

- `wildfire_rl.seeding.set_global_seed(seed)` seeds Python, NumPy, PyTorch, and SB3, and
  sets `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`, deterministic cuDNN, and
  `torch.use_deterministic_algorithms(True, warn_only=True)`.
- The environment dynamics draw from `self.np_random` (seeded by `reset(seed=...)`), **not**
  the global `np.random`. Same seed + same actions ⇒ identical trajectories
  (`tests/test_env_api.py::test_determinism_same_seed`).
- `scripts/check_seed_integrity.py` fails if independent seeds map to byte-identical checkpoints,
  or if eval rows are identical **without** distinct-checkpoint evidence (fabricated multi-seed).
  Identical rows from provably-distinct collapsed policies are the honest PPO negative result and warn.

## No-leakage evaluation splits

- Evaluation seeds each episode as `reset(seed=base_seed + scenario_seed_offset + episode_index)`
  with `eval.scenario_seed_offset` (default 100000), so evaluation ignition scenarios are **disjoint**
  from the training reset-seed range under `env.randomize_ignition=true`. All policies (baselines +
  PPO) are evaluated on **identical** seeds, so comparisons are paired.

## Statistical protocol & effective-method gate

- Report mean, **bootstrap 95% CI** (`eval.significance.bootstrap_ci`, 10k resamples), and Cohen's d
  as primary evidence; p-values are secondary and never reported without an effect size. Cohen's d is
  `nan` (undefined), never `0.0`, when within-group variance is zero.
- `scripts/validate_learning_gate.py` is the **effective-method gate**: the *best reported* policy
  (heuristic routing) must beat no-op; single-agent PPO is retained as an honest negative baseline and
  is **never** tuned or relabeled to "win".
- Report tables are generated from committed CSVs by `scripts/build_report_tables.py` (no hand-entered
  numbers); each carries a `source:` CSV + SHA-256 provenance comment.

## Run metadata

Every training run writes `results/runs/<run_id>/manifest.json` with: UTC timestamp, git SHA,
config hash, seed, resolved library versions, and SHA-256 of the state tensor **and** model
artifacts, plus `curve.csv` (ep_rew_mean vs timestep — evidence of whether training improved). This
ties each reported number to the exact code, config, and bytes that produced it.

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
5. **Eval N.** Reported configs use 50 episodes / 5 disjoint seeds; increase for tighter CIs.
