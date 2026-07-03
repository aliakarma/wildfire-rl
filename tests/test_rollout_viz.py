"""Canonical rollout visualization (Phase 15B.5 / 15B.8).

Asserts the palette/legend contract and that a GIF is produced for a synthetic trajectory.
"""

from __future__ import annotations

import numpy as np

from wildfire_rl.viz.rollout import (
    EVACUATING,
    FIRE,
    GRASS,
    POPULATED,
    Palette,
    class_grid,
    render_gif,
)

CANONICAL = ["Grass", "Finished", "Path", "Populated", "Evacuating", "Fire"]


def test_palette_matches_canonical_contract():
    p = Palette.load()
    assert p.names == CANONICAL, f"palette legend drifted from the canonical contract: {p.names}"
    assert len(p.colors) == 6
    assert all(c.startswith("#") for c in p.colors)


def test_class_grid_priority():
    g = 6
    fire = np.zeros((g, g), dtype=np.float32)
    asset = np.zeros((g, g), dtype=np.int32)
    asset[1, 1] = 1  # asset -> Populated
    fire[4, 4] = 1.0  # fire -> Fire (top priority)
    visited = np.zeros((g, g), dtype=bool)
    visited[2, 2] = True  # Path
    ever = fire > 0.1
    grid = class_grid(fire, asset, [(0, 0)], visited, ever)
    assert grid[0, 0] == EVACUATING  # agent
    assert grid[1, 1] == POPULATED  # asset
    assert grid[2, 2] == 2  # Path
    assert grid[4, 4] == FIRE  # fire on top
    assert grid[5, 5] == GRASS  # untouched


def test_render_gif_produces_file(tmp_path):
    frames = []
    for t in range(1, 4):
        grid = np.zeros((6, 6), dtype=np.int32)
        grid[0, t] = FIRE
        frames.append({"t": t, "grid": grid, "fire_total": float(3 - t)})
    out = tmp_path / "rollout.gif"
    render_gif(frames, out, title="test")
    assert out.exists() and out.stat().st_size > 0
