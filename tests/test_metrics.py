"""Metric tests — verify the single canonical threshold behavior."""

from __future__ import annotations

import numpy as np

from wildfire_rl.eval.metrics import (
    burned_cells,
    catastrophe_prevention_score,
    containment_rate,
    episode_return,
    fire_intensity,
    infrastructure_survival_rate,
    protected_critical_assets,
    risk_adjusted_containment,
    summarize,
    weighted_economic_loss,
)
from wildfire_rl.eval.transfer import (
    adaptation_asymmetry,
    cross_domain_gap,
    transfer_robustness_score,
)


def test_burned_cells_threshold():
    fire = np.array([[0.1, 0.6], [0.9, 0.4]], dtype=np.float32)
    state = np.stack([fire] + [np.zeros_like(fire)] * 6)
    assert burned_cells(state, threshold=0.5) == 2
    assert burned_cells(state, threshold=0.2) == 3  # different threshold -> different count


def test_fire_intensity_sums_fire_channel():
    fire = np.ones((3, 3), dtype=np.float32)
    state = np.stack([fire] + [np.full((3, 3), 9.0, dtype=np.float32)] * 6)
    assert fire_intensity(state) == 9.0  # ignores non-fire channels


def test_episode_return():
    assert episode_return([-1.0, -2.0, -3.0]) == -6.0


def test_summarize_mean_std():
    eps = [{"r": 0.0}, {"r": 2.0}]
    out = summarize(eps)
    assert out["r_mean"] == 1.0
    assert out["r_std"] == 1.0


def test_containment_rate():
    def _state(fire2d):
        return np.stack([fire2d] + [np.zeros_like(fire2d)] * 6)

    # No fire remaining -> full containment.
    assert containment_rate(10.0, _state(np.zeros((4, 4), np.float32))) == 1.0
    # Half the initial mass remains -> 0.5.
    half = np.zeros((4, 4), np.float32)
    half[0, 0] = 5.0
    assert containment_rate(10.0, _state(half)) == 0.5
    # More fire than initial (re-ignition) -> clipped to 0.
    assert containment_rate(1.0, _state(np.ones((4, 4), np.float32))) == 0.0


# ----------------------------------------------------- strategic infrastructure metrics (15B.3)
def _fire_state(fire2d):
    return np.stack([fire2d] + [np.zeros_like(fire2d)] * 6)


def _assets():
    at = np.zeros((4, 4), dtype=np.int32)
    at[0, 0] = 1  # refinery (value 10)
    at[3, 3] = 2  # pipeline (value 4)
    return at


def test_isr_and_pca():
    at = _assets()
    clean = _fire_state(np.zeros((4, 4), np.float32))
    assert infrastructure_survival_rate(at, clean) == 1.0
    assert protected_critical_assets(at, clean) == 2
    # refinery burns -> half survive, one protected
    burn = np.zeros((4, 4), np.float32)
    burn[0, 0] = 1.0
    assert infrastructure_survival_rate(at, _fire_state(burn)) == 0.5
    assert protected_critical_assets(at, _fire_state(burn)) == 1
    # no assets -> neutral
    assert infrastructure_survival_rate(np.zeros((4, 4), np.int32), clean) == 1.0


def test_weighted_economic_loss_value_weighted():
    at = _assets()
    values = {1: 10.0, 2: 4.0}
    ref = np.zeros((4, 4), np.float32)
    ref[0, 0] = 1.0  # fire on refinery
    pipe = np.zeros((4, 4), np.float32)
    pipe[3, 3] = 1.0  # fire on pipeline
    assert weighted_economic_loss(at, _fire_state(ref), values) == 10.0
    assert weighted_economic_loss(at, _fire_state(pipe), values) == 4.0


def test_cps_vs_isr_thresholds():
    at = _assets()
    # light fire on refinery: reached (ISR<1) but not detonated (CPS=1)
    light = np.zeros((4, 4), np.float32)
    light[0, 0] = 0.3  # > reach (0.1), < detonation (0.5)
    assert infrastructure_survival_rate(at, _fire_state(light)) == 0.5
    assert catastrophe_prevention_score(at, _fire_state(light)) == 1.0
    # heavy fire: detonated
    heavy = np.zeros((4, 4), np.float32)
    heavy[0, 0] = 0.9
    assert catastrophe_prevention_score(at, _fire_state(heavy)) == 0.5


def test_risk_adjusted_containment():
    crit = np.zeros((4, 4), np.float32)
    crit[0, 0] = 1.0
    init = np.zeros((4, 4), np.float32)
    init[0, 0] = 1.0
    # fully contained on the critical cell -> RAC 1.0
    assert risk_adjusted_containment(crit, _fire_state(init), _fire_state(np.zeros((4, 4)))) == 1.0
    # unchanged -> RAC 0.0
    assert risk_adjusted_containment(crit, _fire_state(init), _fire_state(init)) == 0.0


def test_defending_assets_beats_minimizing_burned():
    """A policy that saves assets beats one that only minimizes burned cells on ISR/WEL/CPS,
    even at equal burned-cell counts."""
    at = _assets()
    values = {1: 10.0, 2: 4.0}
    # both leave exactly one burning cell (equal burned_cells)
    saves_asset = np.zeros((4, 4), np.float32)
    saves_asset[2, 1] = 1.0  # fire on an empty cell (asset defended)
    loses_asset = np.zeros((4, 4), np.float32)
    loses_asset[0, 0] = 1.0  # fire on the refinery
    assert burned_cells(_fire_state(saves_asset), 0.5) == burned_cells(
        _fire_state(loses_asset), 0.5
    )
    assert infrastructure_survival_rate(
        at, _fire_state(saves_asset)
    ) > infrastructure_survival_rate(at, _fire_state(loses_asset))
    assert weighted_economic_loss(at, _fire_state(saves_asset), values) < weighted_economic_loss(
        at, _fire_state(loses_asset), values
    )


def test_transfer_robustness_and_gap():
    # native ISR 0.8, transfer 0.6
    assert abs(transfer_robustness_score(0.8, 0.6) - 0.75) < 1e-9
    assert abs(cross_domain_gap(0.8, 0.6) - 0.2) < 1e-9
    assert abs(adaptation_asymmetry(0.2, 0.05) - 0.15) < 1e-9
    import math

    assert math.isnan(transfer_robustness_score(0.0, 0.6))
