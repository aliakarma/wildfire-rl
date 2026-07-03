# Wildfire-RL — reproducible workflow entrypoints.
# On Windows without `make`, run the underlying commands shown in each target,
# or use `python scripts/<name>.py` directly (see README "PowerShell equivalents").

PYTHON ?= python
CONFIG ?= configs/experiment/multiseed.yaml

.PHONY: help install install-dev install-geo lint format typecheck test \
        train evaluate transfer ablation marl figures tensors manifest \
        strip-notebooks reproduce check-size clean

help:
	@echo "Targets:"
	@echo "  install        - pip install the package (runtime deps)"
	@echo "  install-dev    - install dev + test + lint tooling, set up pre-commit"
	@echo "  install-geo    - install the geospatial preprocessing stack"
	@echo "  lint / format / typecheck / test"
	@echo "  tensors        - build state tensors from channel layers"
	@echo "  train          - PPO multi-seed training       (CONFIG=$(CONFIG))"
	@echo "  evaluate       - evaluate baselines + PPO model"
	@echo "  transfer       - full cross-region transfer matrix"
	@echo "  ablation / marl - ablation study / MARL scaling"
	@echo "  figures        - regenerate figures from results"
	@echo "  manifest       - write sha256 manifests for data/ and models/"
	@echo "  reproduce      - full pipeline: test, train(both regions), evaluate, marl, ablation, transfer, figures, seed-check"
	@echo "  check-size     - fail if staged files exceed 50 MB (git safety)"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

install-geo:
	$(PYTHON) -m pip install -e ".[geo]"

lint:
	ruff check src tests scripts

format:
	black src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy src

test:
	pytest -q

tensors:
	$(PYTHON) scripts/build_tensors.py --region saudi_eastern_province --grid 32
	$(PYTHON) scripts/build_tensors.py --region california --grid 32

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

transfer:
	$(PYTHON) scripts/transfer.py --config configs/experiment/transfer.yaml

ablation:
	$(PYTHON) scripts/run_ablation.py --config configs/experiment/ablation.yaml

marl:
	$(PYTHON) scripts/train_marl.py --config configs/experiment/scaling.yaml

figures:
	$(PYTHON) scripts/make_figures.py

manifest:
	$(PYTHON) scripts/validate_tensors.py
	$(PYTHON) scripts/make_manifest.py

strip-notebooks:
	$(PYTHON) scripts/strip_notebooks.py notebooks

check-size:
	$(PYTHON) scripts/check_repo_size.py --max-mb 50

reproduce: test
	$(PYTHON) scripts/train.py --config configs/experiment/multiseed.yaml
	$(PYTHON) scripts/train.py --config configs/experiment/multiseed_california.yaml
	$(PYTHON) scripts/evaluate.py --config configs/experiment/multiseed.yaml
	$(PYTHON) scripts/evaluate.py --config configs/experiment/multiseed_california.yaml
	$(PYTHON) scripts/run_marl_evaluation.py --timesteps 100000
	$(PYTHON) scripts/run_ablation.py --config configs/experiment/ablation.yaml
	$(PYTHON) scripts/transfer.py --config configs/experiment/transfer.yaml
	$(PYTHON) scripts/make_figures.py
	$(PYTHON) scripts/build_report_tables.py --out docs/paper/_generated_tables.md
	$(PYTHON) scripts/validate_tensors.py
	$(PYTHON) scripts/check_seed_integrity.py
	$(PYTHON) scripts/validate_learning_gate.py
	@echo "Reproduction pipeline complete. See results/ and figures/."

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ results/runs
