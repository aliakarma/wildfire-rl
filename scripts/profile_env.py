#!/usr/bin/env python
"""Profile FireSuppressionEnv speed — the Phase-1 feasibility gate (REMEDIATION_PLAN_V2).

Cell2Fire runs as a subprocess with CSV I/O per step; this is the top practical risk to a
many-seed MARL study, so it is measured *now*. Times resets (subprocess respawn) and steps
separately, then extrapolates to a 5-seed training+eval study.

    python scripts/profile_env.py --map Sub40x40 --episodes 5 --steps 50
    python scripts/profile_env.py --grid 32 --episodes 5      # picks nearest stock map

Writes per-episode timings + summary to results/profile_env.csv and a provenance record
(git SHA, library versions, args) to results/profile_env_meta.json.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

# Stock maps vendored with Firehose (Phase-2 landscapes will register their own sizes).
_STOCK_MAPS = {20: "Sub20x20", 40: "Sub40x40"}


def pick_map(grid: int) -> str:
    size = min(_STOCK_MAPS, key=lambda s: abs(s - grid))
    name = _STOCK_MAPS[size]
    if size != grid:
        print(
            f"note: no stock {grid}x{grid} map; using nearest stock map {name} "
            f"({size}x{size}). Region-specific 32x32 landscapes arrive in Phase 2."
        )
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", dest="fire_map", default=None, help="Instance name (e.g. Sub40x40)")
    ap.add_argument("--grid", type=int, default=None, help="Grid size -> nearest stock map")
    ap.add_argument("--data-dir", default=None, help="Instance root (e.g. data/cell2fire)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50, help="Max agent steps per episode")
    ap.add_argument("--steps-before-sim", type=int, default=20)
    ap.add_argument(
        "--steps-per-action",
        type=int,
        default=1,
        help="Fire periods (sim-minutes) per agent step; 60 = hourly cadence",
    )
    ap.add_argument(
        "--train-steps-per-seed",
        type=int,
        default=1_000_000,
        help="Assumed env steps per seed for the extrapolation",
    )
    ap.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Assumed eval episodes per seed for the extrapolation",
    )
    ap.add_argument("--seeds", type=int, default=5, help="Seeds in the extrapolated study")
    ap.add_argument("--out", default="results/profile_env.csv")
    args = ap.parse_args()

    from wildfire_marl.env.single_agent_env import FireSuppressionEnv
    from wildfire_marl.reproducibility.logging_utils import write_run_metadata

    fire_map = args.fire_map or pick_map(args.grid or 40)
    env_kwargs = {
        "fire_map": fire_map,
        "steps_before_sim": args.steps_before_sim,
        "steps_per_action": args.steps_per_action,
    }
    if args.data_dir:
        env_kwargs["data_dir"] = args.data_dir
    env = FireSuppressionEnv(**env_kwargs)

    rows: list[dict] = []
    reset_times: list[float] = []
    step_times: list[float] = []
    for ep in range(args.episodes):
        t0 = time.perf_counter()
        env.reset(seed=ep)
        t_reset = time.perf_counter() - t0
        reset_times.append(t_reset)

        n, t_steps = 0, 0.0
        for _ in range(args.steps):
            a = int(env.action_space.sample())
            t0 = time.perf_counter()
            _, _, term, trunc, _ = env.step(a)
            t_steps += time.perf_counter() - t0
            n += 1
            if term or trunc:
                break
        step_times.extend([t_steps / n] * n)
        rows.append(
            {
                "episode": ep,
                "map": fire_map,
                "reset_s": round(t_reset, 4),
                "steps": n,
                "mean_step_s": round(t_steps / n, 4),
                "episode_s": round(t_reset + t_steps, 4),
            }
        )
        print(
            f"episode {ep}: reset {t_reset:.3f}s, {n} steps, "
            f"{t_steps / n * 1000:.1f} ms/step, total {t_reset + t_steps:.2f}s"
        )
    env.close()

    mean_step = statistics.mean(step_times)
    mean_reset = statistics.mean(reset_times)
    steps_per_sec = 1.0 / mean_step

    # ---- extrapolation to the planned study (single process, no parallelism) ----------
    ep_len = max(r["steps"] for r in rows)
    train_s_per_seed = (
        args.train_steps_per_seed * mean_step
        + (args.train_steps_per_seed / max(ep_len, 1)) * mean_reset
    )
    eval_s_per_seed = args.eval_episodes * (mean_reset + ep_len * mean_step)
    total_h = args.seeds * (train_s_per_seed + eval_s_per_seed) / 3600

    print("\n=== Phase-1 feasibility gate ===")
    print(f"map={fire_map}  mean step: {mean_step * 1000:.1f} ms  ({steps_per_sec:.1f} steps/s)")
    print(f"mean reset (subprocess respawn): {mean_reset:.3f} s")
    print(
        f"extrapolation: {args.seeds} seeds x ({args.train_steps_per_seed:,} train steps "
        f"+ {args.eval_episodes} eval episodes) ~= {total_h:.1f} h single-process "
        f"(~{total_h / args.seeds:.1f} h/seed; parallel envs divide wall-clock)"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv

    with out.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        w.writerow(
            {
                "episode": "summary",
                "map": fire_map,
                "reset_s": round(mean_reset, 4),
                "steps": sum(r["steps"] for r in rows),
                "mean_step_s": round(mean_step, 4),
                "episode_s": round(total_h, 2),  # extrapolated study hours (see meta)
            }
        )
    write_run_metadata(
        out.with_name("profile_env_meta.json"),
        config_dict=vars(args),
        artifacts={"profile_csv": out},
    )
    print(f"wrote {out} + provenance meta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
