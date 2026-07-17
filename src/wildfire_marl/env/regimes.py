"""Named benchmark regimes for the multi-agent wildfire-suppression environment.

**Why this exists (Week-6 / Phase-1 benchmark revision).** The original benchmark used a
single extreme ERA5 fire day (FFMC ~96, wind ~21 km/h, FWI ~30) with 60-simulated-minute
action steps. In that regime the fire overruns the whole landscape regardless of what agents
do, so *every* policy — learned or not — ties the No-Op baseline (this is exactly why the
paper's method showed no separation). These regimes instead define **suppression-relevant**
conditions in which skilled coordination measurably protects infrastructure:

  * moderated fire weather (scaled wind + milder FFMC, ISI/FWI recomputed via CFFDRS),
  * a threat-biased ignition sampled upwind of the asset cluster (single ignition — Cell2Fire
    lights one cell per fire — jittered per episode for variety),
  * a crew-scale firebreak footprint (``treat_radius``) so a team can hold a line,
  * a moderate action step so the fire evolves at a controllable rate.

``easy``/``medium``/``hard`` add a difficulty axis for the robustness study; ``legacy`` is the
original extreme regime, retained for reproducibility. **Cell2Fire FBP spread physics is never
modified** — only scenario inputs (weather rows, ignition placement) and the wrapper action
footprint change. Calibrated values come from the Phase-1 No-Op-vs-Value-First sweep.
"""

from __future__ import annotations

from typing import Any

from wildfire_marl.env.marl_env import MultiAgentFireEnv

#: Each regime is a set of env overrides. Keys map onto ``MultiAgentFireEnv`` /
#: ``FireSuppressionEnv`` constructor arguments; ``make_marl_env`` supplies the region-specific
#: paths and the shared infrastructure settings. These are the Saudi/base values.
REGIMES: dict[str, dict[str, Any]] = {
    # Original extreme regime (unsuppressible) — kept only for reproducing prior results.
    "legacy": {
        "steps_per_action": 60,
        "wind_scale": 1.0,
        "ffmc": None,
        "ignition_mode": "uniform",
        "treat_radius": 0,
        "ros_cv": 0.0,
        "max_steps": 150,
    },
    # Suppression-relevant default (== medium). Calibrated so the strong reactive baseline
    # cannot fully contain the fire (Local Reactive WEL ~15, not ~0), leaving room for
    # strategic coordination to win.
    "default": {
        "steps_per_action": 30,
        "wind_scale": 0.6,
        "ffmc": 90.0,
        "ignition_mode": "threat",
        "ignition_dist": (6, 12),
        "ignition_arc_deg": 60.0,
        "treat_radius": 1,
        "ros_cv": 0.1,
        "max_steps": 150,
    },
    "easy": {
        "steps_per_action": 30,
        "wind_scale": 0.4,
        "ffmc": 87.0,
        "ignition_mode": "threat",
        "ignition_dist": (8, 14),
        "ignition_arc_deg": 60.0,
        "treat_radius": 1,
        "ros_cv": 0.1,
        "max_steps": 150,
    },
    "medium": {
        "steps_per_action": 30,
        "wind_scale": 0.6,
        "ffmc": 90.0,
        "ignition_mode": "threat",
        "ignition_dist": (6, 12),
        "ignition_arc_deg": 60.0,
        "treat_radius": 1,
        "ros_cv": 0.1,
        "max_steps": 150,
    },
    "hard": {
        "steps_per_action": 30,
        "wind_scale": 0.8,
        "ffmc": 92.0,
        "ignition_mode": "threat",
        "ignition_dist": (4, 10),
        "ignition_arc_deg": 60.0,
        "treat_radius": 1,
        "ros_cv": 0.1,
        "max_steps": 150,
    },
}

#: Region-specific regime overrides. Saudi (petroleum, concentrated assets) uses the base
#: ``REGIMES`` above; California (WUI, scattered assets, different fuels) spreads differently, so
#: its strategic-headroom band sits at a slightly lighter moderation. Values calibrated from the
#: Phase-1 sweep vs the *strong* Local-Reactive baseline (California default wind x0.5 / FFMC 90:
#: No-Op WEL 34 / ISR 0.00, Local Reactive WEL 11 / ISR 0.69 -> reactive helps but cannot
#: contain the fire, leaving room for strategic coordination).
REGION_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "california": {
        "easy": {"wind_scale": 0.35, "ffmc": 87.0, "steps_per_action": 30, "ignition_dist": (8, 14)},
        "default": {
            "wind_scale": 0.5,
            "ffmc": 90.0,
            "steps_per_action": 30,
            "ignition_dist": (6, 12),
        },
        "medium": {
            "wind_scale": 0.5,
            "ffmc": 90.0,
            "steps_per_action": 30,
            "ignition_dist": (6, 12),
        },
        "hard": {"wind_scale": 0.7, "ffmc": 92.0, "steps_per_action": 30, "ignition_dist": (4, 10)},
    },
}

_REGION_MAP = {"saudi": "Saudi", "california": "California"}


def make_marl_env(
    region: str,
    regime: str = "default",
    *,
    data_dir: str = "data/cell2fire",
    num_agents: int = 3,
    crop_size: int = 9,
    coordination_penalty: float = 0.1,
    catastrophe_weight: float = 2.0,
    cascade_prob: float = 0.1,
    reward_cls: type | None = None,
    **overrides: Any,
) -> MultiAgentFireEnv:
    """Construct a ``MultiAgentFireEnv`` for ``region`` under a named ``regime``.

    A single call site so every method (baselines and the proposed model) shares one env
    definition. ``overrides`` win over the regime defaults (e.g. ``num_agents=6`` for scaling
    studies, or ``regime='default', wind_scale=0.4`` for a one-off).
    """
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime '{regime}' (choose from {sorted(REGIMES)})")
    if region.lower() not in _REGION_MAP:
        raise ValueError(f"Unknown region '{region}' (choose from {sorted(_REGION_MAP)})")

    region_key = region.lower()
    map_name = _REGION_MAP[region_key]
    spec = dict(REGIMES[regime])
    if region_key in REGION_OVERRIDES and regime in REGION_OVERRIDES[region_key]:
        spec.update(REGION_OVERRIDES[region_key][regime])
    spec.update(overrides)

    kwargs: dict[str, Any] = {
        "num_agents": num_agents,
        "crop_size": crop_size,
        "coordination_penalty": coordination_penalty,
        "fire_map": map_name,
        "data_dir": data_dir,
        "observe_infra": True,
        "catastrophe_weight": catastrophe_weight,
        "cascade_prob": cascade_prob,
        "infra_dir": f"{data_dir}/{map_name}",
    }
    if reward_cls is not None:
        kwargs["reward_cls"] = reward_cls
    kwargs.update(spec)
    return MultiAgentFireEnv(**kwargs)
