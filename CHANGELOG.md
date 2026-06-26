# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-26

First public, packaged release. Converts the original notebook-only research project into
an installable, reproducible repository.

### Added
- Installable `wildfire_rl` package (`pip install -e .`) with a `wildfire-rl` CLI.
- Single canonical environment (`WildfireEnv`, `MultiAgentWildfireEnv`) replacing six
  duplicated notebook environment classes.
- Single canonical `CustomCNN` (with optional pooling) replacing ~7 copies.
- Centralized metrics, evaluation loop, transfer matrix, and PPO training pipeline.
- OmegaConf config system (`configs/`) — no hardcoded paths or magic numbers.
- Reproducibility tooling: global seeding, per-env `np_random`, run metadata, sha256
  manifests, `make reproduce`.
- Baseline policies (random / no-op) and a **full symmetric** cross-region transfer matrix
  (including the previously missing California→Saudi cell).
- pytest suite, GitHub Actions CI (lint / test / install / large-file guard), pre-commit.
- Docs: architecture, reproducibility, data card, model card, development guide.
- Strict `.gitignore` / `.gitattributes` and a staged-file size guard to prevent
  accidental large-binary commits.

### Changed
- Fire-spread dynamics reimplemented as vectorized, **seeded** functions (the original used
  the unseeded global `np.random`).
- Single canonical "burned cells" threshold (was inconsistently 0.5 vs 0.2 across notebooks).

### Security
- Removed a hardcoded Google Earth Engine project id from the pipeline; credentials now come
  from environment variables (`.env`).

### Notes
- Trained checkpoints and datasets are no longer tracked in git; they are distributed via
  Hugging Face Hub / Zenodo and fetched with `scripts/fetch_models.py`.
