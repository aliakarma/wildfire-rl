"""FIRMS hotspots → ignition points, with the leakage-free train/eval protocol.

Real NASA FIRMS active-fire detections (``data/<region>/raw/firms/``) are mapped onto the
region grid to form the **ignition candidate set**: cells that historically ignited, restricted
to burnable (fuel) cells. Notably, this restriction matters in the Saudi Eastern Province,
where many FIRMS detections are persistent industrial heat sources (gas flares) on barren
ground — those fall on non-fuel cells and are excluded, with counts reported.

Randomized ignition uses **disjoint train/eval seed streams** — the V1 leakage-free protocol:
evaluation episode *k* draws from ``base_seed + scenario_seed_offset + k`` while training
episode *k* draws from ``base_seed + k``, so train and eval ignition sequences never share an
RNG stream (offset default 100000, as in V1's ``EvalConfig.scenario_seed_offset``).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Default offset separating eval reset seeds from train reset seeds (ported from V1).
SCENARIO_SEED_OFFSET = 100_000


def firms_to_cells(
    firms_csv: str | Path,
    roi: tuple[float, float, float, float],
    grid: int,
) -> dict[int, int]:
    """Map FIRMS detections to 0-indexed grid cells. Returns {cell: detection_count}.

    ``roi`` is the data-card lon/lat rectangle ``(lon_min, lat_min, lon_max, lat_max)``;
    row 0 = max latitude (north-up), matching the V1 tensor registration.
    """
    lon_min, lat_min, lon_max, lat_max = roi
    counts: dict[int, int] = {}
    with open(firms_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if not (lon_min <= lon < lon_max and lat_min < lat <= lat_max):
                continue
            col = int((lon - lon_min) / (lon_max - lon_min) * grid)
            row_i = int((lat_max - lat) / (lat_max - lat_min) * grid)
            col = min(max(col, 0), grid - 1)
            row_i = min(max(row_i, 0), grid - 1)
            cell = row_i * grid + col
            counts[cell] = counts.get(cell, 0) + 1
    return counts


def build_candidates(
    firms_csv: str | Path,
    roi: tuple[float, float, float, float],
    grid: int,
    fuel_codes: np.ndarray,
    nonfuel_code: int,
) -> dict:
    """Ignition candidate record: FIRMS cells restricted to burnable cells.

    Returns a JSON-serializable dict with candidate cells, per-cell detection counts
    (sampling weights), and provenance counts (total/excluded detections).
    """
    counts = firms_to_cells(firms_csv, roi, grid)
    flat = np.asarray(fuel_codes).ravel()
    burnable = {c: n for c, n in counts.items() if flat[c] != nonfuel_code}
    excluded = {c: n for c, n in counts.items() if flat[c] == nonfuel_code}
    cells = sorted(burnable)
    return {
        "grid": grid,
        "roi": list(roi),
        "cells": cells,
        "weights": [burnable[c] for c in cells],
        "n_detections_total": int(sum(counts.values())),
        "n_detections_on_nonfuel_excluded": int(sum(excluded.values())),
        "n_cells_excluded_nonfuel": len(excluded),
        "source": str(firms_csv),
    }


@dataclass
class IgnitionSampler:
    """Seeded, leakage-free ignition sampling over the FIRMS candidate set.

    ``sample_train(k)`` and ``sample_eval(k)`` use disjoint seed streams (see module
    docstring). Sampling is detection-count-weighted, so historically fire-prone cells
    ignite more often — the observed FIRMS spatial ignition distribution.
    """

    cells: list[int]
    weights: list[int]
    base_seed: int = 0
    scenario_seed_offset: int = SCENARIO_SEED_OFFSET

    @classmethod
    def from_json(cls, path: str | Path, base_seed: int = 0) -> IgnitionSampler:
        rec = json.loads(Path(path).read_text())
        return cls(cells=rec["cells"], weights=rec["weights"], base_seed=base_seed)

    def _sample(self, seed: int) -> int:
        if not self.cells:
            raise RuntimeError("Empty ignition candidate set")
        rng = np.random.default_rng(seed)
        w = np.asarray(self.weights, dtype=np.float64)
        return int(rng.choice(self.cells, p=w / w.sum()))

    def sample_train(self, episode: int) -> int:
        return self._sample(self.base_seed + episode)

    def sample_eval(self, episode: int) -> int:
        return self._sample(self.base_seed + self.scenario_seed_offset + episode)


def write_ignitions_csv(path: str | Path, cell_0indexed: int, year: int = 1) -> None:
    """Write a Cell2Fire ``Ignitions.csv`` (cells are 1-indexed in Cell2Fire)."""
    Path(path).write_text(f"Year,Ncell\n{year},{int(cell_0indexed) + 1}\n")
