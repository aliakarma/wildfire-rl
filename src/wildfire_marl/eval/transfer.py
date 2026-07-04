"""Cross-region transfer evaluation — the full, symmetric matrix.

The V1 notebooks only ever computed Saudi->Saudi, Saudi->California, California->California
(and one results file had two of three rows hardcoded). This computes the COMPLETE
N x N matrix in a single run, with one metric definition and one RNG regime, so every
cell is comparable and reproducible. For two regions this includes the previously
missing California->Saudi cell.

Ported from V1 (``legacy_v1/src/wildfire_rl/eval/transfer.py``) with one Phase-0 change:
``transfer_matrix`` no longer imports the retired V1 env-coupled evaluation loop —
the evaluation function is injected as an argument instead. The scoring functions
(TRS / CDGG / adaptation asymmetry) are unchanged. Re-anchored to the Cell2Fire
evaluation loop in Phase 9.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd


def transfer_robustness_score(native: float, transfer: float) -> float:
    """TRS = transfer / native (Phase 15B.3). 1.0 = no degradation; <1 = the metric dropped under
    transfer. For higher-is-better metrics (ISR/CPS/RAC). Returns ``nan`` when ``native`` is ~0."""
    if native is None or transfer is None or abs(native) < 1e-12:
        return float("nan")
    return float(transfer) / float(native)


def cross_domain_gap(native: float, transfer: float) -> float:
    """CDGG = native − transfer (Phase 15B.3): absolute generalization loss (positive = worse under
    transfer for higher-is-better metrics)."""
    if native is None or transfer is None:
        return float("nan")
    return float(native) - float(transfer)


def adaptation_asymmetry(cdgg_a_to_b: float, cdgg_b_to_a: float) -> float:
    """|CDGG(A→B) − CDGG(B→A)|: how unequal the two transfer directions are (Phase 15B.4)."""
    return abs(float(cdgg_a_to_b) - float(cdgg_b_to_a))


def transfer_matrix(
    policies: dict[str, Any],
    env_factories: dict[str, Callable[[], Any]],
    evaluate_fn: Callable[..., dict[str, Any]],
    n_episodes: int = 20,
    base_seed: int = 0,
    deterministic: bool = True,
    scenario_seed_offset: int = 0,
) -> pd.DataFrame:
    """Evaluate every (train policy) x (test environment) combination.

    Args:
        policies: ``{train_region: policy}``.
        env_factories: ``{test_region: env_factory}``.
        evaluate_fn: the canonical evaluation loop (Phase 4+). Must return a dict with a
            ``"summary"`` mapping of aggregated per-episode metrics.

    Returns:
        Tidy DataFrame with one row per (train, test) pair.
    """
    rows: list[dict[str, Any]] = []
    for train_region, policy in policies.items():
        for test_region, factory in env_factories.items():
            result = evaluate_fn(
                policy,
                factory,
                n_episodes=n_episodes,
                base_seed=base_seed,
                deterministic=deterministic,
                scenario_seed_offset=scenario_seed_offset,
            )
            s = result["summary"]
            rows.append(
                {
                    "train_region": train_region,
                    "test_region": test_region,
                    "mean_reward": s.get("episode_reward_mean"),
                    "reward_std": s.get("episode_reward_std"),
                    "mean_burned_cells": s.get("burned_cells_mean"),
                    "mean_fire_intensity": s.get("fire_intensity_mean"),
                    "n_episodes": n_episodes,
                }
            )
    return pd.DataFrame(rows)
