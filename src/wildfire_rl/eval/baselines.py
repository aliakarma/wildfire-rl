"""Baseline control policies.

The original "baseline_evaluation" notebook contained NO baseline — only the PPO model
evaluated on its own region. Without a control there is no evidence the learned policy
beats doing nothing. These policies expose the same ``predict(obs, deterministic)``
interface as Stable-Baselines3 models, so they drop straight into
:func:`wildfire_rl.eval.evaluate.evaluate_policy`.
"""

from __future__ import annotations

import numpy as np

# Action 4 == "stay" (no movement); see wildfire_rl.envs.base.
STAY_ACTION = 4


class RandomPolicy:
    """Uniformly random actions. The lower-bound control for 'did the agent learn?'."""

    def __init__(self, action_space, seed: int | None = None) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = True):  # noqa: ARG002
        if hasattr(self.action_space, "nvec"):  # MultiDiscrete (MARL)
            action = np.array([self.rng.integers(0, n) for n in self.action_space.nvec])
        else:
            action = int(self.rng.integers(0, self.action_space.n))
        return action, None


class NoOpPolicy:
    """Agent never moves (constant 'stay'). Note: local suppression still applies."""

    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def predict(self, obs, deterministic: bool = True):  # noqa: ARG002
        if hasattr(self.action_space, "nvec"):
            action = np.full(len(self.action_space.nvec), STAY_ACTION, dtype=int)
        else:
            action = STAY_ACTION
        return action, None
