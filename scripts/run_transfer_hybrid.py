#!/usr/bin/env python
"""Phase 15B.4 — symmetric, infrastructure-aware cross-region transfer (evaluation-only).

Evaluates each deterministic policy family (`nearest_fire`, `frontier`, `hierarchical_greedy`,
`hierarchical_risk_aware`) on **both** regions with the strategic infrastructure metrics
(ISR / WEL / CPS / RAC), bootstrap 95% CIs over disjoint eval-seed batches, and cross-region
**TRS / CDGG** per metric. This covers both directions (incl. the previously-missing California
coverage) for the *effective methods*.

Scientific note: heuristic/hybrid families are **region-agnostic** (no per-region training), so the
transfer signal is the change in the *fire regime* (desert ↔ forest) and *asset layout*, not policy
specialization. TRS/CDGG therefore measure how well the effective method's asset protection
generalizes across regimes. (Region-specialized transfer — e.g. the trained-PPO negative baseline —
is a separate, heavier study.)

    python scripts/run_transfer_hybrid.py --config configs/experiment/transfer_hybrid.yaml
    python scripts/run_transfer_hybrid.py --config configs/experiment/transfer_hybrid.yaml --seeds 2 --episodes 10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import _bootstrap  # noqa: F401
from wildfire_rl.config import EnvConfig, InfraConfig
from wildfire_rl.coordination.strategic_controller import StrategicController
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv
from wildfire_rl.eval.baselines import FrontierPolicy, NearestFirePolicy
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.significance import bootstrap_ci
from wildfire_rl.eval.transfer import cross_domain_gap, transfer_robustness_score
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.paths import ensure_dir, region_tensor_path, results_dir

logger = get_logger("run_transfer_hybrid")

GRID = 32
STRATEGIC = ["isr", "cps", "rac", "wel"]  # higher better except WEL (lower better)
HIGHER_BETTER = {"isr": True, "cps": True, "rac": True, "wel": False, "burned_cells": False}


def _env_factory(rc, num_agents: int, reward_mode: str):
    tensor = np.load(region_tensor_path(rc.dir, GRID))

    def factory():
        cfg = EnvConfig(
            reward_mode=reward_mode,
            spread_scale=float(rc.spread_scale),
            ignition_rate=float(rc.ignition_rate),
            n_ignition_points=int(rc.n_ignition_points),
            randomize_ignition=True,
        )
        cfg.infra = InfraConfig(
            infra_dir=rc.infra_dir,
            catastrophe_weight=float(rc.catastrophe_weight),
            cascade_prob=float(rc.cascade_prob),
        )
        return MultiAgentWildfireEnv(state_tensor=tensor, config=cfg, num_agents=num_agents)

    return factory


def _make_policy(family: str, sample_env):
    a, g = sample_env.action_space, sample_env.grid_size
    if family == "nearest_fire":
        return NearestFirePolicy(a, grid_size=g, env=sample_env)
    if family == "frontier":
        return FrontierPolicy(a, grid_size=g, env=sample_env)
    if family == "hierarchical_greedy":
        return StrategicController(a, grid_size=g, variant="greedy_risk", env=sample_env)
    if family == "hierarchical_risk_aware":
        return StrategicController(a, grid_size=g, variant="risk_aware", env=sample_env)
    raise ValueError(f"unknown family {family!r}")


def _evaluate_cell(policy, factory, seeds, n_episodes, offset):
    """Return {metric: (mean, ci_lo, ci_hi)} aggregated over per-seed-batch means."""
    per_seed: dict[str, list[float]] = {}
    for s in seeds:
        summary = evaluate_policy(
            policy, factory, n_episodes=n_episodes, base_seed=int(s), scenario_seed_offset=offset
        )["summary"]
        for key in ("burned_cells", *STRATEGIC):
            mk = f"{key}_mean"
            if mk in summary:
                per_seed.setdefault(key, []).append(float(summary[mk]))
    out: dict[str, tuple[float, float, float]] = {}
    for key, vals in per_seed.items():
        lo, hi = bootstrap_ci(vals)
        out[key] = (float(np.mean(vals)), lo, hi)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment/transfer_hybrid.yaml")
    ap.add_argument("--seeds", type=int, default=None, help="Override: use first N seeds.")
    ap.add_argument("--episodes", type=int, default=None, help="Override episodes/seed.")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    seeds = list(cfg.seeds)[: args.seeds] if args.seeds else list(cfg.seeds)
    n_episodes = args.episodes or int(cfg.n_episodes)
    offset = int(cfg.scenario_seed_offset)
    families = list(cfg.families)
    regions = dict(cfg.regions)

    # 1. Evaluate every (family x region) cell.
    cells: dict[tuple[str, str], dict] = {}
    rows = []
    for region_name, rc in regions.items():
        factory = _env_factory(rc, int(cfg.num_agents), str(cfg.reward_mode))
        sample_env = factory()
        for family in families:
            policy = _make_policy(family, sample_env)
            logger.info(
                "Evaluating %s on %s (%d seeds x %d eps)",
                family,
                region_name,
                len(seeds),
                n_episodes,
            )
            res = _evaluate_cell(policy, factory, seeds, n_episodes, offset)
            cells[(family, region_name)] = res
            row = {
                "policy_family": family,
                "region": region_name,
                "reward_mode": str(cfg.reward_mode),
                "n_seeds": len(seeds),
                "n_episodes": n_episodes,
            }
            for key, (m, lo, hi) in res.items():
                row[f"{key}_mean"], row[f"{key}_ci_lo"], row[f"{key}_ci_hi"] = m, lo, hi
            rows.append(row)

    res_dir = ensure_dir(results_dir())
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "transfer_hybrid.csv", index=False)
    logger.info("Wrote %s (%d cells)", res_dir / "transfer_hybrid.csv", len(df))

    # 2. Cross-region generalization per (family x metric): native = each region as anchor.
    region_names = list(regions.keys())
    gen_rows = []
    if len(region_names) == 2:
        a, b = region_names
        for family in families:
            for metric in ("burned_cells", *STRATEGIC):
                va = cells[(family, a)].get(metric, (float("nan"),))[0]
                vb = cells[(family, b)].get(metric, (float("nan"),))[0]
                gen_rows.append(
                    {
                        "policy_family": family,
                        "metric": metric,
                        "higher_is_better": HIGHER_BETTER.get(metric, True),
                        f"{a}_mean": va,
                        f"{b}_mean": vb,
                        "trs_a_to_b": transfer_robustness_score(va, vb),  # b relative to a-native
                        "cdgg_a_to_b": cross_domain_gap(va, vb),
                        "trs_b_to_a": transfer_robustness_score(vb, va),
                        "cdgg_b_to_a": cross_domain_gap(vb, va),
                    }
                )
        gdf = pd.DataFrame(gen_rows)
        gdf.to_csv(res_dir / "transfer_hybrid_generalization.csv", index=False)
        logger.info("Wrote %s (%d rows)", res_dir / "transfer_hybrid_generalization.csv", len(gdf))

    # console summary: ISR per family per region
    print("\n=== Infrastructure Survival Rate (ISR) — family x region ===")
    for family in families:
        vals = " | ".join(
            f"{r}={cells[(family, r)].get('isr', (float('nan'),))[0]:.3f}" for r in region_names
        )
        print(f"  {family:26s} {vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
