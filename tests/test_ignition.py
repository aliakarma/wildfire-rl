"""FIRMS ignition sampler tests (Phase 2) — leakage-free protocol properties."""

from __future__ import annotations

import numpy as np

from wildfire_marl.data.ignition import (
    IgnitionSampler,
    build_candidates,
    firms_to_cells,
    write_ignitions_csv,
)

ROI = (45.0, 23.0, 50.0, 28.0)  # lon_min, lat_min, lon_max, lat_max


def _write_firms(tmp_path, rows):
    p = tmp_path / "firms.csv"
    hdr = "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type"
    body = [
        f"{lat},{lon},340,0.5,0.5,2025-06-01,0900,SNPP,SNPP,n,2,300,5.0,D,0" for lat, lon in rows
    ]
    p.write_text("\n".join([hdr] + body) + "\n")
    return p


def test_firms_to_cells_northup_registration(tmp_path):
    # max-lat detection -> row 0; min-lon -> col 0
    p = _write_firms(tmp_path, [(27.9, 45.1), (23.1, 49.9)])
    counts = firms_to_cells(p, ROI, grid=4)
    assert counts == {0: 1, 15: 1}  # NW corner cell and SE corner cell


def test_candidates_exclude_nonfuel_cells(tmp_path):
    """Gas-flare-style detections on non-fuel land must be excluded."""
    p = _write_firms(tmp_path, [(27.9, 45.1), (27.9, 45.1), (23.1, 49.9)])
    codes = np.full((4, 4), 31)  # all grass...
    codes[3, 3] = 101  # ...except the SE corner (non-fuel)
    rec = build_candidates(p, ROI, 4, codes, nonfuel_code=101)
    assert rec["cells"] == [0]
    assert rec["weights"] == [2]
    assert rec["n_detections_on_nonfuel_excluded"] == 1


def test_train_eval_streams_disjoint_and_deterministic():
    s = IgnitionSampler(cells=list(range(50)), weights=[1] * 50, base_seed=0)
    train = [s.sample_train(k) for k in range(20)]
    evalv = [s.sample_eval(k) for k in range(20)]
    # deterministic
    assert train == [s.sample_train(k) for k in range(20)]
    assert evalv == [s.sample_eval(k) for k in range(20)]
    # different streams (identical sequences would mean shared RNG -> leakage)
    assert train != evalv


def test_weighted_sampling_prefers_hot_cells():
    s = IgnitionSampler(cells=[7, 8], weights=[1000, 1], base_seed=0)
    draws = [s.sample_train(k) for k in range(200)]
    assert draws.count(7) > 180


def test_write_ignitions_csv_is_one_indexed(tmp_path):
    out = tmp_path / "Ignitions.csv"
    write_ignitions_csv(out, cell_0indexed=798)
    assert out.read_text() == "Year,Ncell\n1,799\n"
