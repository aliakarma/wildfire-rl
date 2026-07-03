# Reproducibility

This project targets artifact-evaluation-grade reproducibility. This document is the
protocol a reviewer should follow, and the honest list of current limitations.

## Certification status (Phase 14)

**Local certification — PASS.** On the current tree every gate is green and every committed artifact
hash-verifies against its manifest:

| Check | Result |
|---|---|
| Artifact integrity (`models_manifest.json`) | 162/162 verified, 0 changed/missing |
| Artifact integrity (`data_manifest.json`) | 40/40 verified, 0 changed/missing |
| `validate_tensors.py` | exit 0 (shape `(7,H,W)`, all channels `[0,1]`) |
| `check_seed_integrity.py` | exit 0 (34 distinct checkpoints; California collapse is the honest negative result, not fraud) |
| `validate_learning_gate.py` | exit 0 (effective-method gate: heuristic ≪ no-op; PPO negative result) |
| Report ↔ CSV | `build_report_tables.py` idempotent; §16 tables carry `source:` + SHA-256 |
| Provenance | 10 run manifests + 10 training curves; 25 figures regenerated from CSVs |
| Tests / lint | `pytest` 72 passed; `black`+`ruff` clean; CI `validate` job blocking |

Expected reference hashes (SHA-256, first 16 hex): `data_manifest.json` `bcfaa5df4236ecd5`,
`models_manifest.json` `9f8970c0195dab1a`, `ppo_saudi_32_seed_0.zip` `4a2e7237b3d0aca0`,
`ppo_california_32_seed_0.zip` `3a576f47627ca5e2`, `saudi/.../state_tensor.npy` `57ee7d48e52f6275`.
Full transcript: `results/runs/CERTIFICATION_*.log`.

> **`make reproduce` is not re-run for certification.** It retrains every model from scratch; on CPU
> that is ~10 h of undertrained runs that would overwrite the authoritative Colab-T4 checkpoints. The
> checkpoints are the authoritative artifacts, and the downstream `evaluate → figures → tables → gates`
> path reproduces deterministically **from** them.

**External (third-party) clean-clone certification — PENDING owner action:** (1) commit + push this
tree; (2) upload `models/*.zip` to the Hub repo referenced by `scripts/fetch_models.py`; (3) on a clean
clone run `pip install -r requirements.lock.txt`, `fetch_models.py --verify results/models_manifest.json`,
`validate_tensors.py`, `check_seed_integrity.py`, `validate_learning_gate.py`, `pytest -q`.

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
