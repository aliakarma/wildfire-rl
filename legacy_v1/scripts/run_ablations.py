#!/usr/bin/env python
"""Phase 15B.8 — strategic ablation runner (evaluation-only, deterministic families).

Enumerates the ablation groups (`hybrid_vs_pure`, `infra_reward`, `strategic_components`,
`catastrophe`), evaluates each cell with the strategic metrics (ISR/WEL/CPS/RAC/PA/CCL) + coordination
efficiency (CE), bootstrap 95% CIs over disjoint eval-seed batches, writes a tidy provenance CSV per
group under `results/ablation/`, and renders a canonical rollout GIF per cell.

PPO is retained only as the honest negative baseline (see `results/eval_saudi.csv`); it is not re-run.

    python scripts/run_ablations.py --group all
    python scripts/run_ablations.py --group hybrid_vs_pure --seeds 2 --episodes 8
    python scripts/run_ablations.py --group all --render-figures
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from wildfire_rl.ablation import GROUPS, build_env_factory, make_cell_policy
from wildfire_rl.eval import metrics
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.significance import bootstrap_ci
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.paths import ensure_dir, figures_dir, results_dir
from wildfire_rl.viz.rollout import record_rollout, render_gif

logger = get_logger("run_ablations")

METRIC_KEYS = ["burned_cells", "isr", "wel", "cps", "rac", "pa", "ccl"]
NUM_AGENTS = 5


def _coordination_efficiency(cell, seed, offset, max_steps=60):
    """One rollout tracking agent-cell distinctness per step -> CE."""
    factory = build_env_factory(cell, NUM_AGENTS)
    env = factory()
    policy = make_cell_policy(cell, env)
    obs, _ = env.reset(seed=seed + offset)
    policy.env = env
    steps = []
    for _ in range(max_steps):
        steps.append([tuple(p) for p in env.agent_positions])
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    return metrics.coordination_efficiency(steps)


def _run_cell(cell, seeds, n_episodes, offset):
    factory = build_env_factory(cell, NUM_AGENTS)
    sample_env = factory()
    policy = make_cell_policy(cell, sample_env)
    per_seed: dict[str, list[float]] = {}
    for s in seeds:
        summary = evaluate_policy(
            policy, factory, n_episodes=n_episodes, base_seed=int(s), scenario_seed_offset=offset
        )["summary"]
        for key in METRIC_KEYS:
            mk = f"{key}_mean"
            if mk in summary:
                per_seed.setdefault(key, []).append(float(summary[mk]))
    agg = {}
    for key, vals in per_seed.items():
        lo, hi = bootstrap_ci(vals)
        agg[key] = (float(np.mean(vals)), lo, hi)
    agg["ce"] = (_coordination_efficiency(cell, int(seeds[0]), offset), float("nan"), float("nan"))
    return agg


def run_group(group: str, seeds, n_episodes, offset, render: bool):
    cells = GROUPS[group]
    res_dir = ensure_dir(results_dir() / "ablation")
    fig_dir = ensure_dir(figures_dir() / "ablation" / group)
    rows = []
    for cell_name, cell in cells.items():
        logger.info("[%s] cell=%s", group, cell_name)
        agg = _run_cell(cell, seeds, n_episodes, offset)
        gif_path = fig_dir / f"{cell_name}.gif"
        if render:
            factory = build_env_factory(cell, NUM_AGENTS)
            env = factory()
            frames = record_rollout(env, make_cell_policy(cell, env), seed=offset, max_steps=50)
            render_gif(frames, gif_path, title=f"{group} · {cell_name}")
        row = {
            "group": group,
            "cell": cell_name,
            "controller": cell["controller"],
            "variant": cell.get("variant"),
            "coordinate": cell["coordinate"],
            "catastrophe_weight": cell["catastrophe_weight"],
            "cascade_prob": cell["cascade_prob"],
            "n_seeds": len(seeds),
            "n_episodes": n_episodes,
            "rollout_gif": f"figures/ablation/{group}/{cell_name}.gif" if render else "",
        }
        for key, (m, lo, hi) in agg.items():
            row[f"{key}_mean"], row[f"{key}_ci_lo"], row[f"{key}_ci_hi"] = m, lo, hi
        rows.append(row)
    df = pd.DataFrame(rows)
    out = res_dir / f"{group}.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d cells)", out, len(df))
    # console: ISR + RAC per cell
    print(f"\n=== {group}: ISR / RAC / CE per cell ===")
    for _, r in df.iterrows():
        print(
            f"  {r['cell']:26s} ISR={r.get('isr_mean', float('nan')):.3f}  "
            f"RAC={r.get('rac_mean', float('nan')):+.3f}  CE={r.get('ce_mean', float('nan')):.3f}"
        )
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", default="all", help="all | " + " | ".join(GROUPS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--offset", type=int, default=100000)
    ap.add_argument("--render-figures", action="store_true")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    groups = list(GROUPS) if args.group == "all" else [args.group]
    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group {g!r}; known: {list(GROUPS)}")
        run_group(g, seeds, args.episodes, args.offset, args.render_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
