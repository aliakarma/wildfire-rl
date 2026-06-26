#!/usr/bin/env python
"""Download trained PPO checkpoints from the Hugging Face Hub into ``models/``.

Checkpoints are NOT stored in git (they total ~2.2 GB). They are published to the Hub
and verified against a checksum manifest. Set ``HF_TOKEN`` for private repos.

    python scripts/fetch_models.py --repo aliakarma/wildfire-rl-ppo
    python scripts/fetch_models.py --repo aliakarma/wildfire-rl-ppo --verify results/models_manifest.json
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.paths import ensure_dir, models_dir

logger = get_logger("fetch_models")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="HF Hub repo id, e.g. user/wildfire-rl-ppo")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--verify", default=None, help="Optional checksum manifest to verify against.")
    args = ap.parse_args()

    dest = ensure_dir(models_dir())
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install -e .[track]")
        return 1

    logger.info("Downloading %s@%s -> %s", args.repo, args.revision, dest)
    snapshot_download(
        repo_id=args.repo, revision=args.revision, local_dir=str(dest),
        local_dir_use_symlinks=False, allow_patterns=["*.zip", "*.json"],
    )

    if args.verify:
        from wildfire_rl.data.manifest import verify_manifest

        problems = verify_manifest(dest, args.verify)
        if problems:
            logger.error("Checksum mismatches:\n%s", "\n".join(problems))
            return 2
        logger.info("All checkpoints verified against %s", args.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
