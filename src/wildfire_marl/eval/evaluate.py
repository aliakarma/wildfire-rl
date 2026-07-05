"""Canonical evaluation loop for single-agent fire-suppression policies (Phase 4).

Features disjoint evaluation seeds (no training leakage) and aggregates
both standard and critical-infrastructure strategic metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from wildfire_marl.eval import metrics


class Policy(Protocol):
    def predict(
        self,
        obs: np.ndarray,
        action_masks: np.ndarray | None = None,
        deterministic: bool = ...,
    ) -> tuple[Any, Any]: ...


def _strategic_metrics(
    env: Any, initial_state: np.ndarray, final_state: np.ndarray
) -> dict[str, float]:
    """Calculate per-episode infrastructure metrics if the env has assets."""
    asset_type = getattr(env, "asset_type", None)
    criticality = getattr(env, "criticality", None)
    if asset_type is None:
        return {}

    asset_values = getattr(env, "asset_values", {1: 10.0, 2: 4.0, 3: 6.0, 4: 3.0})
    cascade = int(getattr(env, "cascade_ignited_count", 0))

    return {
        "isr": metrics.infrastructure_survival_rate(asset_type, final_state),
        "pca": float(metrics.protected_critical_assets(asset_type, final_state)),
        "wel": metrics.weighted_economic_loss(asset_type, final_state, asset_values),
        "cps": metrics.catastrophe_prevention_score(asset_type, final_state),
        "rac": metrics.risk_adjusted_containment(criticality, initial_state, final_state),
        "pa": metrics.prioritization_accuracy(asset_type, final_state, asset_values),
        "ccl": metrics.catastrophe_chain_length(cascade, asset_type, final_state),
    }


def evaluate_policy(
    policy: Any,
    env_factory: Callable[[], Any],
    n_episodes: int = 20,
    base_seed: int = 0,
    deterministic: bool = True,
    burned_threshold: float = 0.5,
    scenario_seed_offset: int = 100000,
) -> dict[str, Any]:
    """Evaluate a policy over n_episodes with explicit, disjoint seeds.

    Args:
        policy: Heuristic policy or trained SB3 model.
        env_factory: Zero-arg factory function returning a fresh env.
        n_episodes: Number of episodes to run.
        base_seed: Base seed for the random stream.
        deterministic: Whether to use deterministic prediction.
        burned_threshold: Threshold to consider a cell burned.
        scenario_seed_offset: Offset added to reset seeds to prevent training leakage.

    Returns:
        results: Dictionary containing 'summary' (means and stds) and 'episodes' list.
    """
    per_episode: list[dict[str, float]] = []

    for ep in range(n_episodes):
        env = env_factory()
        # Seed logic: base_seed + offset + ep
        seed = base_seed + scenario_seed_offset + ep
        obs, _ = env.reset(seed=seed)

        initial_state = env.fire_state.copy()
        initial_fire_total = float(np.sum(initial_state > 0)) or 1.0

        done = False
        rewards: list[float] = []

        while not done:
            # Check for action masks (required for MaskablePPO)
            action_masks = None
            if hasattr(env, "action_masks"):
                action_masks = env.action_masks()

            if action_masks is not None:
                # Try passing action_masks parameter
                try:
                    action, _ = policy.predict(
                        obs, action_masks=action_masks, deterministic=deterministic
                    )
                except TypeError:
                    action, _ = policy.predict(obs, deterministic=deterministic)
            else:
                action, _ = policy.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            done = bool(terminated or truncated)

        final_state = env.fire_state.copy()

        # Build standard metrics record
        record = {
            "episode_reward": metrics.episode_return(rewards),
            "burned_cells": metrics.burned_cells(final_state, burned_threshold),
            "fire_intensity": metrics.fire_intensity(final_state),
            "containment_rate": metrics.containment_rate(initial_fire_total, final_state),
        }

        # Add strategic infrastructure metrics if present
        record.update(_strategic_metrics(env, initial_state, final_state))

        per_episode.append(record)
        env.close()

    summary = metrics.summarize(per_episode)
    summary["n_episodes"] = float(n_episodes)
    summary["burned_threshold"] = burned_threshold

    return {"summary": summary, "episodes": per_episode}
