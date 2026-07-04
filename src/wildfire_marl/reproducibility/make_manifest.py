#!/usr/bin/env python
"""Write sha256 manifests for data/ and models/ (reproducibility snapshots).

Ported from V1 (``legacy_v1/scripts/make_manifest.py``) in Phase 0; logic unchanged,
imports re-pointed at ``wildfire_marl``.

    python -m wildfire_marl.reproducibility.make_manifest
"""

from __future__ import annotations

from wildfire_marl.paths import data_dir, ensure_dir, models_dir, results_dir
from wildfire_marl.reproducibility.logging_utils import get_logger
from wildfire_marl.reproducibility.manifest import write_manifest

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
