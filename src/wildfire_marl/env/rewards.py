"""Pluggable reward functions for the Cell2Fire suppression environments.

The env computes state; a ``Reward`` object maps (env state) -> scalar. Any subclass can be
swapped in via ``FireSuppressionEnv(reward_cls=...)`` — this is the hook Phase 3 uses to add
``InfrastructureWeightedReward`` without touching env or simulator code.

``FireSizeReward`` is ported from Firehose (``third_party/firehose/cell2fire/firehose/rewards.py``)
so our baseline objective is identical to the published benchmark's.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wildfire_marl.env.single_agent_env import FireSuppressionEnv


class Reward(ABC):
    """A reward function over the env's current state (the env updates state first)."""

    def __init__(self, env: "FireSuppressionEnv"):
        self.env = env

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, action: int | list[int] | None = None) -> float:
        raise NotImplementedError


class FireSizeReward(Reward):
    """-(cells on fire) / (total cells) * scale — the Firehose benchmark objective."""

    def __init__(self, env: "FireSuppressionEnv", scale: float = 10.0):
        super().__init__(env)
        self.scale = scale

    @classmethod
    def name(cls) -> str:
        return "FireSizeReward"

    def __call__(self, action: int | list[int] | None = None) -> float:
        state = self.env.fire_state  # (H, W): >0 on fire, <0 harvested, 0 untouched
        num_on_fire = int(np.sum(state > 0))
        return -num_on_fire / self.env.num_cells * self.scale


class InfrastructureWeightedReward(Reward):
    """Phase-3 hook: fire-size penalty PLUS value-weighted damage on critical assets.

    Requires the criticality/asset rasters built in Phase 3
    (``wildfire_marl.infra.build_infrastructure``). Declared here so the reward interface is
    frozen in Phase 1; instantiating it before Phase 3 lands is an explicit error, not a
    silent zero.
    """

    @classmethod
    def name(cls) -> str:
        return "InfrastructureWeightedReward"

    def __call__(self, action: int | list[int] | None = None) -> float:
        raise NotImplementedError(
            "InfrastructureWeightedReward arrives in Phase 3 (real asset rasters + values). "
            "Use FireSizeReward until then."
        )


REWARD_CLASSES: dict[str, type[Reward]] = {
    cls.name(): cls for cls in (FireSizeReward, InfrastructureWeightedReward)
}
