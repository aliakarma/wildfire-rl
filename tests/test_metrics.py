"""Metric tests — verify the single canonical threshold behavior."""

from __future__ import annotations

import numpy as np

from wildfire_rl.eval.metrics import burned_cells, episode_return, fire_intensity, summarize


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
