# Development Guide

## Setup

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

## Common tasks

| Task | Command |
|---|---|
| Format | `make format` (black + ruff --fix) |
| Lint | `make lint` |
| Type-check | `make typecheck` |
| Test | `make test` or `pytest -q` |
| Coverage | `pytest --cov=wildfire_rl --cov-report=term-missing` |
| Size guard | `make check-size` |
| Strip notebook outputs | `make strip-notebooks` |

## Repository invariants (enforced in review + CI)

- One env, one CNN, one metrics module, one training loop. No notebook copies.
- No hardcoded paths (`wildfire_rl.paths`) or magic numbers (`configs/`).
- Stochasticity uses `self.np_random` / explicit `Generator`.
- No file > 50 MB committed (CI `large-files` job + pre-commit `check-added-large-files`).
- Notebook outputs stripped (pre-commit `nbstripout`).

## Testing strategy

- `test_env_api.py` — Gymnasium API conformance + determinism + truncation.
- `test_tensor_stack.py` — stacking shape/order + normalization.
- `test_metrics.py` — canonical thresholds.
- `test_config.py` — defaults + dotlist overrides.
- `test_seeding.py` — reproducible RNG.
- `test_eval_baselines.py` — baseline policies + evaluation loop (torch-free).
- `test_cli_smoke.py` — CLI dispatch.

Heavy training is **not** in the default suite; mark slow/integration tests with the
`@pytest.mark.slow` / `@pytest.mark.integration` markers (configured in `pyproject.toml`).

## Releasing

1. Update `CHANGELOG.md` and bump `version` in `pyproject.toml` + `CITATION.cff`.
2. Tag: `git tag -a v0.1.0 -m "v0.1.0"` (push tags when ready).
3. Publish checkpoints to Hugging Face; archive a snapshot on Zenodo for a DOI.
4. Add the DOI badge to `README.md` and `doi:` to `CITATION.cff`.

## Project conventions

- Python ≥ 3.10, line length 100, `from __future__ import annotations`.
- First-party import root: `wildfire_rl` (configured in ruff isort).
- Logging via `wildfire_rl.logging_utils.get_logger`, not `print`.
