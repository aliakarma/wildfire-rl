"""Canonical evaluation loop.

One implementation, used by training, baseline, and transfer evaluation alike. Key fix
vs. the notebooks: every episode is reset with an explicit, distinct seed, so evaluation
is reproducible *and* (when the env has ``randomize_ignition=True``) actually samples a
distribution of held-out scenarios rather than re-running one fixed map.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from wildfire_rl.config import MetricsConfig
from wildfire_rl.eval import metrics


class Policy(Protocol):
    def predict(self, obs, deterministic: bool = ...): ...


def evaluate_policy(
    policy: Policy,
    env_factory: Callable[[], Any],
    n_episodes: int = 20,
    base_seed: int = 0,
    deterministic: bool = True,
    metrics_cfg: MetricsConfig | None = None,
) -> dict[str, Any]:
    """Run ``n_episodes`` and return aggregated + per-episode metrics.

    Args:
        policy: anything with ``predict(obs, deterministic) -> (action, state)``.
        env_factory: zero-arg callable returning a fresh env each episode.
        base_seed: episode ``i`` uses ``reset(seed=base_seed + i)``.
    """
    metrics_cfg = metrics_cfg or MetricsConfig()
    per_episode: list[dict[str, float]] = []

    for ep in range(n_episodes):
        env = env_factory()
        if hasattr(policy, "env"):
            policy.env = env
        if hasattr(policy, "set_env"):
            policy.set_env(env)
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        rewards: list[float] = []
        while not done:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            done = bool(terminated or truncated)

        final_state = env.state
        per_episode.append(
            {
                "episode_reward": metrics.episode_return(rewards),
                "burned_cells": metrics.burned_cells(final_state, metrics_cfg.burned_threshold),
                "fire_intensity": metrics.fire_intensity(final_state),
            }
        )

    summary = metrics.summarize(per_episode)
    summary["n_episodes"] = n_episodes
    summary["burned_threshold"] = metrics_cfg.burned_threshold
    return {"summary": summary, "episodes": per_episode}
