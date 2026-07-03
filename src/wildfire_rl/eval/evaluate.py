"""Canonical evaluation loop.

One implementation, used by training, baseline, and transfer evaluation alike. Key fix
vs. the notebooks: every episode is reset with an explicit, distinct seed, so evaluation
is reproducible *and* (when the env has ``randomize_ignition=True``) actually samples a
distribution of held-out scenarios rather than re-running one fixed map.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from wildfire_rl.config import MetricsConfig
from wildfire_rl.eval import metrics


class Policy(Protocol):
    def predict(self, obs, deterministic: bool = ...): ...


def _strategic_metrics(env, initial_state, final_state) -> dict[str, float]:
    """Per-episode infrastructure metrics (Phase 15B.3), emitted only when the env has assets.

    Returns ISR / WEL / CPS / RAC / PCA when ``env`` carries infrastructure rasters, else ``{}`` so
    non-infrastructure evaluations are unchanged.
    """
    asset_type = getattr(env, "infra_asset_type", None)
    infra_crit = getattr(env, "infra_criticality", None)
    # Prefer the infra criticality; fall back to the Phase-15 reward criticality if present.
    criticality = infra_crit if infra_crit is not None else getattr(env, "criticality", None)
    if asset_type is None:
        return {}
    asset_values = dict(env.cfg.infra.asset_values)
    cascade = int(getattr(env, "_cascade_ignited_total", 0))
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
    policy: Policy,
    env_factory: Callable[[], Any],
    n_episodes: int = 20,
    base_seed: int = 0,
    deterministic: bool = True,
    metrics_cfg: MetricsConfig | None = None,
    scenario_seed_offset: int = 0,
) -> dict[str, Any]:
    """Run ``n_episodes`` and return aggregated + per-episode metrics.

    Args:
        policy: anything with ``predict(obs, deterministic) -> (action, state)``.
        env_factory: zero-arg callable returning a fresh env each episode.
        base_seed: episode ``i`` uses ``reset(seed=base_seed + scenario_seed_offset + i)``.
        scenario_seed_offset: offset added to every reset seed so evaluation scenarios are
            disjoint from the training reset seeds (no train/test leakage under
            ``randomize_ignition``). Callers pass ``cfg.eval.scenario_seed_offset``.
    """
    metrics_cfg = metrics_cfg or MetricsConfig()
    per_episode: list[dict[str, float]] = []

    for ep in range(n_episodes):
        env = env_factory()
        if hasattr(policy, "env"):
            policy.env = env
        if hasattr(policy, "set_env"):
            policy.set_env(env)
        obs, _ = env.reset(seed=base_seed + scenario_seed_offset + ep)
        initial_state = np.asarray(env.state).copy()
        done = False
        rewards: list[float] = []
        while not done:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            done = bool(terminated or truncated)

        final_state = env.state
        record = {
            "episode_reward": metrics.episode_return(rewards),
            "burned_cells": metrics.burned_cells(final_state, metrics_cfg.burned_threshold),
            "fire_intensity": metrics.fire_intensity(final_state),
            "containment_rate": metrics.containment_rate(
                getattr(env, "_initial_fire_total", 1.0), final_state
            ),
        }
        record.update(_strategic_metrics(env, initial_state, final_state))
        per_episode.append(record)

    summary = metrics.summarize(per_episode)
    summary["n_episodes"] = n_episodes
    summary["burned_threshold"] = metrics_cfg.burned_threshold
    return {"summary": summary, "episodes": per_episode}
