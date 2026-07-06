"""Canonical metric definitions — the single source of truth.

The notebooks counted "burned cells" with threshold ``> 0.5`` in training but ``> 0.2``
in evaluation, so numbers were not comparable across files. Here the threshold is an
explicit parameter (default 0.5, from :class:`MetricsConfig`) and every caller uses it.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BURNED_THRESHOLD = 0.5


def burned_cells(state: np.ndarray, threshold: float = DEFAULT_BURNED_THRESHOLD) -> int:
    """Number of cells whose fire intensity exceeds ``threshold``.

    Accepts a full ``(C, H, W)`` state tensor or a 2D fire channel.
    """
    fire = state[0] if state.ndim == 3 else state
    return int(np.sum(fire > threshold))


def fire_intensity(state: np.ndarray) -> float:
    """Total fire intensity (sum over the fire channel)."""
    fire = state[0] if state.ndim == 3 else state
    return float(np.sum(fire))


def containment_rate(initial_fire_total: float, final_state: np.ndarray) -> float:
    """Fraction of the initial fire mass no longer burning at episode end (higher = better).

    An agent-influenceable outcome metric, unlike raw ``fire_intensity`` (dominated by fire the
    single agent cannot reach). Clipped to ``[0, 1]``.
    """
    fire = final_state[0] if final_state.ndim == 3 else final_state
    final = float(np.sum(fire))
    denom = initial_fire_total or 1.0
    return float(min(1.0, max(0.0, 1.0 - final / denom)))


def episode_return(rewards: list[float]) -> float:
    return float(np.sum(rewards))


# --------------------------------------------------------------------------------------------------
# Strategic infrastructure metrics (Phase 15B.3)
#
# Assets are cells with ``asset_type > 0``. "reached" = fire above ``reach_threshold`` (0.1);
# "detonated" = fire above ``detonation_threshold`` (0.5). ``damage_a`` for a cell is its final fire
# intensity in [0, 1]. All functions return neutral values (1.0 / 0.0) when there are no assets.
# --------------------------------------------------------------------------------------------------

REACH_THRESHOLD = 0.1
DETONATION_THRESHOLD = 0.5


def _fire2d(state: np.ndarray) -> np.ndarray:
    return state[0] if state.ndim == 3 else state


def infrastructure_survival_rate(
    asset_type: np.ndarray, final_state: np.ndarray, reach_threshold: float = REACH_THRESHOLD
) -> float:
    """ISR = fraction of assets never reached by fire (final fire ≤ ``reach_threshold``)."""
    if asset_type is None:
        return 1.0
    mask = asset_type > 0
    total = int(mask.sum())
    if total == 0:
        return 1.0
    fire = _fire2d(final_state)
    survived = int(((fire <= reach_threshold) & mask).sum())
    return survived / total


def protected_critical_assets(
    asset_type: np.ndarray, final_state: np.ndarray, reach_threshold: float = REACH_THRESHOLD
) -> int:
    """PCA = absolute count of assets with no fire damage (final fire ≤ ``reach_threshold``)."""
    if asset_type is None:
        return 0
    mask = asset_type > 0
    fire = _fire2d(final_state)
    return int(((fire <= reach_threshold) & mask).sum())


def weighted_economic_loss(
    asset_type: np.ndarray, final_state: np.ndarray, asset_values: dict[int, float]
) -> float:
    """WEL = Σ_asset-cells value[type] · final_fire (risk-weighted loss; lower is better)."""
    if asset_type is None:
        return 0.0
    fire = _fire2d(final_state)
    total = 0.0
    for code, val in asset_values.items():
        m = asset_type == int(code)
        if m.any():
            total += float(val) * float(fire[m].sum())
    return float(total)


def catastrophe_prevention_score(
    asset_type: np.ndarray,
    final_state: np.ndarray,
    detonation_threshold: float = DETONATION_THRESHOLD,
) -> float:
    """CPS = 1 − detonated_assets / total_assets (avoided catastrophic detonations).

    Distinct from ISR: an asset can be *reached* by light fire yet not *detonate*. CPS uses the
    higher detonation threshold, ISR the lower reach threshold.
    """
    if asset_type is None:
        return 1.0
    mask = asset_type > 0
    total = int(mask.sum())
    if total == 0:
        return 1.0
    fire = _fire2d(final_state)
    detonated = int(((fire > detonation_threshold) & mask).sum())
    return 1.0 - detonated / total


def prioritization_accuracy(
    asset_type: np.ndarray,
    final_state: np.ndarray,
    asset_values: dict[int, float],
    reach_threshold: float = REACH_THRESHOLD,
) -> float:
    """PA = protected asset value / total asset value (Phase 15B.8). 1.0 = the full high-value set
    defended. Value-weighted, so saving a refinery counts more than saving a pipeline."""
    if asset_type is None:
        return 1.0
    fire = _fire2d(final_state)
    total_val = protected_val = 0.0
    for code, val in asset_values.items():
        m = asset_type == int(code)
        n = int(m.sum())
        if n == 0:
            continue
        total_val += float(val) * n
        protected_val += float(val) * int(((fire <= reach_threshold) & m).sum())
    return protected_val / total_val if total_val > 0 else 1.0


def catastrophe_chain_length(
    cascade_ignited: int,
    asset_type: np.ndarray,
    final_state: np.ndarray,
    detonation_threshold: float = DETONATION_THRESHOLD,
) -> float:
    """CCL = cascade-ignited cells / detonated assets (Phase 15B.8): mean chain depth per detonation
    (lower is better; 0 when nothing detonated)."""
    if asset_type is None:
        return 0.0
    fire = _fire2d(final_state)
    detonated = int(((fire > detonation_threshold) & (asset_type > 0)).sum())
    if detonated == 0:
        return 0.0
    return float(cascade_ignited) / detonated


def coordination_efficiency(agent_step_positions: list[list[tuple[int, int]]]) -> float:
    """CE = fraction of agent-steps where agents occupy DISTINCT cells (Phase 15B.8): 1.0 = no
    redundant dispatch (agents never pile on the same cell); lower = wasted/duplicated effort.

    ``agent_step_positions`` is a list (per timestep) of the agents' (x, y) cells.
    """
    if not agent_step_positions:
        return 1.0
    total = distinct = 0
    for positions in agent_step_positions:
        n = len(positions)
        if n == 0:
            continue
        total += n
        distinct += len(set(map(tuple, positions)))
    return distinct / total if total else 1.0


def emergency_response_latency(first_reached_step: list[int], arrival_step: list[int]) -> float:
    """ERL = mean(arrival − threat-onset) over threatened assets (Phase 15B.8); lower is better.

    ``first_reached_step[a]`` = step fire first threatened asset a (or -1 if never); ``arrival_step[a]``
    = step an agent first reached asset a (or the horizon if never). Assets never threatened are
    excluded; when none were threatened, returns 0.0.
    """
    lat = [
        max(0, arrival_step[a] - first_reached_step[a])
        for a in range(len(first_reached_step))
        if first_reached_step[a] >= 0
    ]
    return float(np.mean(lat)) if lat else 0.0


def transfer_robustness_gap(trs_values: list[float]) -> float:
    """TRG = 1 − min(TRS) over the off-diagonal transfer cells (Phase 15B.8): worst-case degradation.
    NaN TRS values are ignored; 0.0 when none are finite."""
    finite = [v for v in trs_values if v == v]  # drop NaN
    return float(1.0 - min(finite)) if finite else 0.0


def risk_adjusted_containment(
    criticality: np.ndarray, initial_state: np.ndarray, final_state: np.ndarray
) -> float:
    """RAC = 1 − Σ(crit·fire_final) / Σ(crit·fire_initial) (criticality-weighted containment).

    1.0 = all criticality-weighted fire contained; ≤0 = fire grew on critical cells.
    """
    if criticality is None:
        return containment_rate(fire_intensity(initial_state), final_state)
    fi = _fire2d(initial_state)
    ff = _fire2d(final_state)
    denom = float((criticality * fi).sum())
    if denom <= 0.0:
        total_crit = float(criticality.sum())
        if total_crit <= 0.0:
            return 1.0
        return float(1.0 - (criticality * ff).sum() / total_crit)
    return float(1.0 - (criticality * ff).sum() / denom)


def summarize(episodes: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-episode metric dicts into mean/std for each key."""
    if not episodes:
        return {}
    keys = episodes[0].keys()
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([e[k] for e in episodes], dtype=float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    return out
