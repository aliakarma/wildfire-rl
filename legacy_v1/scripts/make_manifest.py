#!/usr/bin/env python
"""Write sha256 manifests for data/ and models/ (reproducibility snapshots).

    python scripts/make_manifest.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from wildfire_rl.data.manifest import write_manifest
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.paths import data_dir, ensure_dir, models_dir, results_dir

logger = get_logger("make_manifest")


def main() -> int:
    res = ensure_dir(results_dir())
    if data_dir().exists():
        out = write_manifest(data_dir(), res / "data_manifest.json")
        logger.info("Wrote %s", out)
    if models_dir().exists():
        out = write_manifest(models_dir(), res / "models_manifest.json")
        logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
