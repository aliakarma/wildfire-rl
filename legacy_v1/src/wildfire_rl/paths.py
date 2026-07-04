"""Centralized path resolution.

Replaces the hardcoded ``/content/drive/MyDrive/PyroRL_Saudi_Project/...`` paths
that were scattered across every notebook. All roots are resolved relative to the
repository root and can be overridden with environment variables, so the same code
runs unchanged on Colab, a workstation, or CI.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root.

    Resolution order:
      1. ``$WILDFIRE_REPO_ROOT`` if set.
      2. Walk up from this file until a directory containing ``pyproject.toml`` is
         found (works for editable installs and source checkouts).
      3. Fall back to the current working directory.
    """
    env = os.environ.get("WILDFIRE_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


def _root(env_var: str, default_subdir: str) -> Path:
    env = os.environ.get(env_var)
    return Path(env).expanduser().resolve() if env else repo_root() / default_subdir


def data_dir() -> Path:
    """Root for datasets (raw/processed/grids). Override with ``$WILDFIRE_DATA_DIR``."""
    return _root("WILDFIRE_DATA_DIR", "data")


def models_dir() -> Path:
    """Root for model checkpoints. Override with ``$WILDFIRE_MODELS_DIR``."""
    return _root("WILDFIRE_MODELS_DIR", "models")


def results_dir() -> Path:
    """Root for result CSVs / logs. Override with ``$WILDFIRE_RESULTS_DIR``."""
    return _root("WILDFIRE_RESULTS_DIR", "results")


def figures_dir() -> Path:
    """Root for generated figures. Override with ``$WILDFIRE_FIGURES_DIR``."""
    return _root("WILDFIRE_FIGURES_DIR", "figures")


def configs_dir() -> Path:
    return repo_root() / "configs"


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create ``path`` (and parents) if missing; return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def region_tensor_path(region_dir: str, grid_size: int, filename: str = "state_tensor.npy") -> Path:
    """Resolve the canonical state-tensor path for a region/grid.

    Example: ``region_tensor_path("saudi_eastern_province", 32)`` ->
    ``<data>/saudi_eastern_province/grids/32x32/state_tensor.npy``.
    """
    return data_dir() / region_dir / "grids" / f"{grid_size}x{grid_size}" / filename
