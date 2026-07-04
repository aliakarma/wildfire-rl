#!/usr/bin/env python
"""Phase-2 plausibility smoke: free-burning fire on a converted region landscape.

Runs the interactive Cell2Fire binary (no suppression actions) on a region's generated
landscape and checks the physics is *plausible*, not just runnable:

  1. fire grows monotonically while fuel and weather allow (burned-fraction curve);
  2. spread drifts downwind — the fire-centroid displacement bearing is compared against
     the mean Weather.csv WD over the simulated window (also validates the WD convention:
     a 180° error would show up as anti-alignment);
  3. low-fuel self-extinguish — a second fire ignited at the most fuel-poor burnable cell
     must end far smaller than the main run.

Writes results/sim_smoke_<region>.json (metrics, tracked) and figures/sim_smoke_<region>.gif
+ _final.png (renders, regenerable). One agent-free step = 60 fire periods = 1 simulated hour.

    python scripts/sim_smoke.py --region saudi
    python scripts/sim_smoke.py --region california
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

_REGION_DIRS = {"saudi": "Saudi", "california": "California"}

# fuel-code -> RGB for rendering (matches the standard lookup's intent)
_CODE_COLORS = {
    101: (130, 130, 130),  # non-fuel grey
    31: (255, 255, 190),  # O-1a pale straw
    32: (230, 230, 110),  # O-1b straw
    7: (170, 210, 120),  # C-7 light green
    5: (90, 170, 90),  # C-5 mid green
    3: (25, 110, 45),  # C-3 dark green
}
_FIRE = (255, 30, 0)
_IGNITION = (255, 0, 255)


def _load_codes(map_dir: Path) -> np.ndarray:
    return np.loadtxt(map_dir / "Forest.asc", skiprows=6, dtype=int)


def _frame(codes: np.ndarray, fire: np.ndarray, ignition: int | None, scale: int = 12):
    h, w = codes.shape
    im = np.zeros((h, w, 3), dtype=np.uint8)
    for code, rgb in _CODE_COLORS.items():
        im[codes == code] = rgb
    im[fire > 0] = _FIRE
    if ignition is not None:
        im[divmod(ignition, w)] = _IGNITION
    return np.kron(im, np.ones((scale, scale, 1), dtype=np.uint8))


def _bearing_deg(drow: float, dcol: float) -> float:
    """Compass bearing of a grid displacement (row down = south, col right = east)."""
    return (math.degrees(math.atan2(dcol, -drow)) + 360.0) % 360.0


def _ang_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _run_fire(map_dir: Path, ignition_cell: int, hours: int, burn_in_min: int = 60):
    """Free-burn `hours` simulated hours; return per-hour fire grids."""
    from wildfire_marl.env.cell2fire_binding import Cell2FireBinding, read_grid_csv

    inp = Path(tempfile.mkdtemp(prefix="smoke_in_"))
    shutil.copytree(map_dir, inp, dirs_exist_ok=True)
    (inp / "Ignitions.csv").write_text(f"Year,Ncell\n1,{ignition_cell + 1}\n")
    binding = Cell2FireBinding(inp, seed=123, steps_before_sim=burn_in_min, steps_per_action=60)
    grids: list[np.ndarray] = []
    csvs = binding.restart()
    if csvs:
        grids.append(read_grid_csv(csvs[-1]))
    for _ in range(hours):
        binding.apply_actions(None)  # no suppression
        csvs = binding.progress_to_next_state()
        if csvs:
            grids.append(read_grid_csv(csvs[-1]))
        if binding.finished:
            break
    binding.close()
    shutil.rmtree(inp, ignore_errors=True)
    return grids


def _mean_downwind(map_dir: Path, hours: int) -> float:
    """Circular-mean spread-push azimuth: Weather.csv WD is the FROM-direction, and
    Cell2Fire pushes fire toward WD+180 (ReadCSV.cpp)."""
    rows = (map_dir / "Weather.csv").read_text().splitlines()[1 : hours + 1]
    wds = [(float(r.split(",")[6]) + 180.0) % 360.0 for r in rows]
    s = sum(math.sin(math.radians(w)) for w in wds)
    c = sum(math.cos(math.radians(w)) for w in wds)
    return (math.degrees(math.atan2(s, c)) + 360.0) % 360.0


def _fuel_poor_cell(codes: np.ndarray, candidates: list[int]) -> tuple[int, float]:
    """Burnable candidate with the highest non-fuel share in its radius-2 neighbourhood.

    Returns (cell, nf_share). On a near-uniform fuel carpet (Saudi grass) no isolated pocket
    may exist — the reported nf_share makes that legible instead of silently comparing
    apples to oranges.
    """
    w = codes.shape[1]
    best, best_share = candidates[0], -1.0
    for cell in candidates:
        y, x = divmod(cell, w)
        nb = codes[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
        share = float(np.mean(nb == 101))
        if share > best_share:
            best, best_share = cell, share
    return best, best_share


def _interior_cell(codes: np.ndarray, candidates: list[int], weights: list[int]) -> int:
    """Highest-weighted candidate away from grid edges (for the wind-drift check).

    Drift vs wind is only a valid physics check while the fire is free of map-boundary and
    non-fuel-coast effects; an edge ignition (e.g. Saudi's Gulf-coast default cell) measures
    fuel geometry, not wind response.
    """
    h, w = codes.shape
    margin = max(4, h // 5)
    scored = [
        (weights[i], c)
        for i, c in enumerate(candidates)
        if margin <= c // w < h - margin and margin <= c % w < w - margin
    ]
    if not scored:  # fall back: maximize distance to the nearest edge
        return max(
            candidates,
            key=lambda c: min(c // w, h - 1 - c // w, c % w, w - 1 - c % w),
        )
    return max(scored)[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True, choices=sorted(_REGION_DIRS))
    ap.add_argument("--hours", type=int, default=48, help="Simulated hours of free burn")
    args = ap.parse_args()

    region_dir = _REGION_DIRS[args.region]
    map_dir = Path("data/cell2fire") / region_dir
    codes = _load_codes(map_dir)
    h, w = codes.shape
    report = json.loads((map_dir / "conversion_report.json").read_text())
    cand = json.loads((map_dir / "ignition_candidates.json").read_text())
    ignition = int(report["ignition"]["default_cell"])

    # ---- main free burn (default FIRMS ignition; burn curve + renders) -------------------
    grids = _run_fire(map_dir, ignition, args.hours)
    burned = [int((g > 0).sum()) for g in grids]
    frac = [round(b / codes.size, 4) for b in burned]

    # ---- wind-drift check: interior ignition, EARLY window (pre-boundary) ----------------
    drift_ign = _interior_cell(codes, cand["cells"], cand["weights"])
    drift_hours = min(12, args.hours)
    dgrids = _run_fire(map_dir, drift_ign, drift_hours)
    c0 = np.argwhere(dgrids[0] > 0).mean(axis=0)
    c1 = np.argwhere(dgrids[-1] > 0).mean(axis=0)
    drift_bearing = _bearing_deg(float(c1[0] - c0[0]), float(c1[1] - c0[1]))
    wd = _mean_downwind(map_dir, len(dgrids))
    misalign = _ang_diff(drift_bearing, wd)

    # ---- low-fuel self-extinguish check ---------------------------------------------------
    poor_cell, nf_share = _fuel_poor_cell(codes, cand["cells"])
    poor_grids = _run_fire(map_dir, poor_cell, args.hours)
    poor_final = int((poor_grids[-1] > 0).sum())

    metrics = {
        "region": args.region,
        "hours_simulated": len(grids) - 1,
        "ignition_cell": ignition,
        "burned_cells_curve_hourly": burned,
        "final_burned_fraction": frac[-1],
        "monotone_growth": bool(all(b2 >= b1 for b1, b2 in zip(burned, burned[1:], strict=False))),
        "drift_ignition_cell": int(drift_ign),
        "drift_window_hours": len(dgrids) - 1,
        "drift_bearing_deg": round(drift_bearing, 1),
        "mean_downwind_azimuth_deg": round(wd, 1),
        "drift_wind_misalignment_deg": round(misalign, 1),
        "wind_aligned_within_90deg": bool(misalign < 90.0),
        "lowfuel_ignition_cell": int(poor_cell),
        "lowfuel_neighbourhood_nf_share": round(nf_share, 3),
        "lowfuel_final_burned": poor_final,
        "lowfuel_vs_main_ratio": round(poor_final / max(burned[-1], 1), 4),
    }
    out_json = Path("results") / f"sim_smoke_{args.region}.json"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))

    # ---- renders ------------------------------------------------------------------------
    from PIL import Image

    figs = Path("figures")
    figs.mkdir(exist_ok=True)
    frames = [
        Image.fromarray(_frame(codes, g, ignition)) for g in grids[:: max(1, len(grids) // 60)]
    ]
    frames[0].save(
        figs / f"sim_smoke_{args.region}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
    )
    Image.fromarray(_frame(codes, grids[-1], ignition)).save(
        figs / f"sim_smoke_{args.region}_final.png"
    )

    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_json}, figures/sim_smoke_{args.region}.gif/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
