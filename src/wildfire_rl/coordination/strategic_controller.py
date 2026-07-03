"""Strategic high-level controller over a heuristic low level (Phase 15B.2).

The honest negative result (§1.2) motivates a hierarchical hybrid design: PPO fails at low-level
navigation, but a heuristic router is robust and near-optimal there. So we isolate the *learnable
strategic* problem (which sector/asset to defend, and which agent goes where) from the
*un-learnable-here navigation* problem (how to get there), which the deterministic router solves.

``StrategicController`` is an env-agnostic :class:`Policy` (drop-in for ``evaluate_policy``): it reads
the environment's fire field, infrastructure rasters, and agent positions; ranks sectors by risk;
assigns each agent to a sector; and routes it toward that sector's top target with the low-level
router. The three variants are directly comparable because they share this contract:

  * ``greedy_risk`` — rank sectors by raw fire load (argmax risk).
  * ``risk_aware``  — rank by fire load + ``infra_risk_weight`` × fire-on-criticality (defends assets).
  * ``rl``          — rank by an injected learned ``scorer`` over the small strategic action space
    (falls back to ``greedy_risk`` when no scorer is provided; training the high level is 15B.4/future).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wildfire_rl.config import HierarchyConfig
from wildfire_rl.routing.frontier_router import get_frontier_target
from wildfire_rl.routing.nearest_fire_router import get_nearest_fire_target
from wildfire_rl.routing.routing_utils import compute_step_action, get_sector_bounds

VARIANTS = ("greedy_risk", "risk_aware", "rl")


class StrategicController:
    """High-level strategic dispatcher over a robust low-level heuristic router."""

    uses_privileged_state = True  # reads env asset rasters + agent positions (oracle localization)

    def __init__(
        self,
        action_space,
        grid_size: int = 32,
        variant: str = "greedy_risk",
        num_sectors: int = 4,
        low_level: str = "nearest_fire",
        infra_risk_weight: float = 5.0,
        spread_threshold: float = 0.2,
        coordinate: bool = True,
        env=None,
        scorer: Callable[[int, np.ndarray, np.ndarray | None], float] | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.action_space = action_space
        self.grid_size = grid_size
        self.variant = variant
        self.num_sectors = num_sectors
        self.low_level = low_level
        self.infra_risk_weight = infra_risk_weight
        self.spread_threshold = spread_threshold
        self.coordinate = coordinate  # False = ablate coordination (all agents pile on top sector)
        self.env = env
        self.scorer = scorer

    # ------------------------------------------------------------------ high level
    def _sector_scores(self, fire: np.ndarray, criticality: np.ndarray | None) -> list[float]:
        scores: list[float] = []
        for s in range(self.num_sectors):
            rs, cs = get_sector_bounds(s, self.grid_size)
            fire_load = float(fire[rs, cs].sum())
            if self.variant == "risk_aware" and criticality is not None:
                infra_risk = float((criticality[rs, cs] * fire[rs, cs]).sum())
                score = fire_load + self.infra_risk_weight * infra_risk
            elif self.variant == "rl" and self.scorer is not None:
                crit_sec = None if criticality is None else criticality[rs, cs]
                score = float(self.scorer(s, fire[rs, cs], crit_sec))
            else:  # greedy_risk (and rl fallback)
                score = fire_load
            scores.append(score)
        return scores

    # ------------------------------------------------------------------- low level
    def _target(self, fire: np.ndarray, ax: int, ay: int, sector: int) -> tuple[int, int]:
        if self.low_level == "frontier":
            return get_frontier_target(fire, ax, ay, sector, self.spread_threshold)
        return get_nearest_fire_target(fire, ax, ay, sector, self.spread_threshold)

    # ------------------------------------------------------------------- Policy API
    def predict(self, obs, deterministic: bool = True):  # noqa: ARG002
        if self.env is None:
            raise RuntimeError("StrategicController needs `env` set (evaluate_policy sets it).")
        fire = np.asarray(self.env.state[0])
        criticality = getattr(self.env, "infra_criticality", None)
        positions = self.env.agent_positions

        scores = self._sector_scores(fire, criticality)
        ranked = sorted(range(self.num_sectors), key=lambda s: scores[s], reverse=True)

        actions = []
        for i, (ax, ay) in enumerate(positions):
            # coordinate: spread agents across top-priority sectors; else all pile on the top sector
            sector = ranked[i % len(ranked)] if self.coordinate else ranked[0]
            tx, ty = self._target(fire, int(ax), int(ay), sector)
            actions.append(compute_step_action(int(ax), int(ay), int(tx), int(ty), self.grid_size))
        return np.array(actions, dtype=int), None


def make_strategic_controller(
    action_space, cfg: HierarchyConfig | None = None, grid_size: int = 32, env=None, **kwargs
) -> StrategicController:
    """Build a :class:`StrategicController` from a :class:`HierarchyConfig` (or defaults)."""
    cfg = cfg or HierarchyConfig()
    return StrategicController(
        action_space,
        grid_size=grid_size,
        variant=cfg.high_level,
        num_sectors=cfg.num_sectors,
        low_level=cfg.low_level,
        infra_risk_weight=cfg.infra_risk_weight,
        env=env,
        **kwargs,
    )
