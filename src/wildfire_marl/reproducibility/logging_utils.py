"""Logging + run-metadata utilities.

Replaces bare ``print`` calls with a configured logger, and provides a single helper
to capture run metadata (git SHA, config hash, library versions, seed, timestamp) so
every artifact is traceable back to the exact run that produced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONFIGURED = False


def get_logger(name: str = "wildfire_marl", level: int = logging.INFO) -> logging.Logger:
    """Return a module logger, configuring the root handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _lib_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "torch", "gymnasium", "pettingzoo", "stable_baselines3", "pandas"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "n/a"
    return versions


def config_hash(config_dict: dict[str, Any]) -> str:
    """Stable short hash of a config dict (for grouping/comparing runs)."""
    blob = json.dumps(config_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def file_sha256(path: str | Path | None) -> str | None:
    """Full SHA-256 of a file, or ``None`` if the path is unset/missing.

    Binds a run to the exact tensors/checkpoints it consumed.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_metadata(
    config_dict: dict[str, Any] | None = None,
    seed: int | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a reproducibility metadata record for the current run.

    ``artifacts`` maps a name to a file path; each is recorded with its SHA-256 so the run is
    bound to the exact tensors/checkpoints it used.
    """
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "seed": seed,
        "libraries": _lib_versions(),
    }
    if config_dict is not None:
        meta["config_hash"] = config_hash(config_dict)
        meta["config"] = config_dict
    if artifacts:
        meta["artifacts"] = {name: file_sha256(p) for name, p in artifacts.items()}
    return meta


def write_run_metadata(
    path: str | Path,
    config_dict: dict[str, Any] | None = None,
    seed: int | None = None,
    artifacts: dict[str, Any] | None = None,
) -> Path:
    """Write run metadata to ``path`` as JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_metadata(config_dict, seed, artifacts), indent=2))
    return path
