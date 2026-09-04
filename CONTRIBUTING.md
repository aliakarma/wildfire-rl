# Contributing to Wildfire-RL

Thanks for your interest in improving Wildfire-RL! This guide covers the workflow and the
standards CI enforces.

## Development setup

```bash
git clone https://github.com/aliakarma/wildfire-rl.git
cd wildfire-rl
python -m pip install -e ".[dev]"
pre-commit install          # runs ruff/black/nbstripout/large-file checks on commit
```

## Workflow

1. Create a branch off `main`: `git checkout -b feature/short-description`.
2. Make focused changes with tests.
3. Run the local checks (mirrors CI):
   ```bash
   make format      # black + ruff --fix
   make lint        # ruff
   make typecheck   # mypy
   make test        # pytest
   make check-size  # no staged file > 50 MB
   ```
4. Open a PR. Describe *what* changed and *why*; link any issue.

## Standards

- **Single source of truth.** There is exactly one environment (`envs/base.py`,
  `envs/multi_agent.py`), one CNN (`models/cnn.py`), one metrics module (`eval/metrics.py`),
  one training loop (`train/ppo.py`). Do **not** reintroduce per-region/per-notebook copies.
- **No hardcoded paths or magic numbers.** Use `wildfire_marl.paths` and `configs/`.
- **Determinism.** New stochastic code must use the env's `self.np_random` (single-agent /
  MARL) or an explicit `numpy.random.Generator` — never the global `np.random`.
- **Metrics.** Use `eval/metrics.py`; don't invent new thresholds inline.
- **Notebooks.** Keep logic in `src/`; notebooks only orchestrate. Outputs are stripped by
  `nbstripout` (pre-commit) — never commit large embedded outputs.
- **Large files.** Never commit models, raw data, or tensors. Host on Hugging Face / Zenodo
  and add a `fetch_*` path. `.gitignore` + `check_repo_size.py` guard this.

## Adding a new region

1. Place channel layers under `data/<region>/grids/<G>x<G>/` and run
   `python -m wildfire_marl.data.to_cell2fire --region <region> --grid <G>` to emit the
   Cell2Fire landscape under `data/cell2fire/<Region>/`.
2. Register the region in `src/wildfire_marl/env/regimes.py` so `make_marl_env` resolves it.
3. Add a training config under `configs/marl/` modelled on an existing one. No new code.

## Reporting bugs / requesting features

Open an issue with a minimal reproduction (config + command), expected vs. actual behavior,
and your environment (`wildfire-rl info`).
