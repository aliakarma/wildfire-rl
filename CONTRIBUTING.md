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
- **No hardcoded paths or magic numbers.** Use `wildfire_rl.paths` and `configs/`.
- **Determinism.** New stochastic code must use the env's `self.np_random` (single-agent /
  MARL) or an explicit `numpy.random.Generator` — never the global `np.random`.
- **Metrics.** Use `eval/metrics.py`; don't invent new thresholds inline.
- **Notebooks.** Keep logic in `src/`; notebooks only orchestrate. Outputs are stripped by
  `nbstripout` (pre-commit) — never commit large embedded outputs.
- **Large files.** Never commit models, raw data, or tensors. Host on Hugging Face / Zenodo
  and add a `fetch_*` path. `.gitignore` + `check_repo_size.py` guard this.

## Adding a new region

1. Place channel layers under `data/<region>/grids/<G>x<G>/` and run
   `python scripts/build_tensors.py --region <region> --grid <G>`.
2. Add `configs/region/<region>.yaml`.
3. Reference it via `--config` / `--set region.dir=<region> region.name=<name>`. No new code.

## Reporting bugs / requesting features

Open an issue with a minimal reproduction (config + command), expected vs. actual behavior,
and your environment (`wildfire-rl info`).
