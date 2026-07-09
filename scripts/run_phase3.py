"""Phase-3 run harness: train every method on the suppression-relevant regime, evaluate all
under the seed-mean protocol, and emit the results tables.

Idempotent: a learned method whose checkpoint already exists is not retrained (use ``--force``
to override), so the run resumes cleanly across Colab sessions. One training seed (42) per
method per region; evaluation over disjoint seed groups x episodes (paper protocol).

Examples
--------
Smoke (validate end-to-end, minutes)::

    python scripts/run_phase3.py --quick

Full run (both regions, all methods)::

    python scripts/run_phase3.py --train-steps 100000 --episodes 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from wildfire_marl.env.regimes import make_marl_env  # noqa: E402
from wildfire_marl.env.rewards import WELISRDeltaReward  # noqa: E402
from wildfire_marl.eval.phase3_eval import (  # noqa: E402
    ALL_KINDS,
    LABELS,
    LEARNED,
    checkpoint_path,
    evaluate,
)
from wildfire_marl.eval.significance import bootstrap_ci, welch_ttest  # noqa: E402
from wildfire_marl.train.hier_comm_train import train_hier_comm  # noqa: E402
from wildfire_marl.train.marl_train import (  # noqa: E402
    set_seed,
    train_commnet,
    train_mappo,
    train_qmix,
)

METRICS = ["WEL", "ISR", "CE", "burned", "return"]


def _base_cfg(kind: str, steps: int, quick: bool) -> dict:
    """Training config per method (Phase-3 defaults; trimmed in --quick)."""
    if kind in ("mappo", "commnet", "qmix"):
        cfg = {
            "total_steps": steps,
            "n_steps": 1024,
            "batch_size": 64 if kind != "qmix" else 32,
            "ppo_epochs": 4,
            "clip_eps": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "lr": 5e-4,
            "comm_rounds": 2,
            "target_update_interval": 200,
        }
    else:  # hiercomm / hiercomm_heur
        cfg = {
            "total_steps": steps,
            "n_steps": 1500,
            "ppo_epochs": 4,
            "batch_size": 256,
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "clip_eps": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.02,
            "shaping_coef": 0.05,
            "comm_rounds": 2,
            "bc_tactical_episodes": 6 if quick else 30,
            "bc_tactical_epochs": 20 if quick else 60,
            "pretrain_episodes": 6 if quick else 40,
            "bc_epochs": 30 if quick else 100,
            "rl_lr": 5e-5,
            "kl_coef": 0.3,
            "cmd_ent_coef": 0.05,
            "commander": "learned" if kind == "hiercomm" else "heuristic",
        }
    return cfg


def train_method(kind: str, region: str, steps: int, quick: bool, out_dir: Path, device) -> None:
    """Train one learned method (seed 42) and save the checkpoint in the Phase-3 convention."""
    set_seed(42)
    env = make_marl_env(region, "default", reward_cls=WELISRDeltaReward)
    cfg = _base_cfg(kind, steps, quick)
    t0 = time.time()
    print(f"\n[TRAIN] {kind} / {region} ({steps} steps)...", flush=True)
    ckpt: dict = {"config": cfg}
    curve = None
    if kind == "mappo":
        actor, _critic, curve = train_mappo(env, cfg, device)
        ckpt["actor_state_dict"] = actor.state_dict()
    elif kind == "commnet":
        actor, _critic, curve = train_commnet(env, cfg, device)
        ckpt["actor_state_dict"] = actor.state_dict()
    elif kind == "qmix":
        agent, mixer, curve = train_qmix(env, cfg, device)
        ckpt["agent_state_dict"] = agent.state_dict()
        ckpt["mixer_state_dict"] = mixer.state_dict()
    else:  # hiercomm / hiercomm_heur
        actor, _critic, commander, curve = train_hier_comm(env, cfg, device)
        ckpt["actor_state_dict"] = actor.state_dict()
        if commander is not None:
            ckpt["commander_state_dict"] = commander.state_dict()
    env.close()
    path = checkpoint_path(kind, region, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    if curve:
        pd.DataFrame(curve).to_csv(out_dir / f"train_curve_{path.stem}.csv", index=False)
    print(
        f"[TRAIN] {kind}/{region} done in {(time.time() - t0) / 60:.1f} min -> {path}", flush=True
    )


def seed_means(
    records: list[dict], policy: str, region: str, metric: str, seeds: list[int]
) -> np.ndarray:
    """Per-seed-group mean of a metric for one policy (unit of statistical analysis)."""
    out = []
    for s in seeds:
        vals = [
            r[metric]
            for r in records
            if r["policy"] == policy and r["region"] == region and r["seed"] == s
        ]
        if vals:
            out.append(float(np.mean(vals)))
    return np.asarray(out, dtype=float)


def build_summary(
    records: list[dict], regions: list[str], kinds: list[str], seeds: list[int]
) -> dict:
    """Seed-mean summary + key Welch comparisons (vs No-Op, Value-First, HierComm)."""
    summary: dict = {"protocol": {"seeds": seeds, "unit": "seed means"}, "regions": {}}
    for region in regions:
        reg: dict = {}
        for kind in kinds:
            reg[kind] = {"label": LABELS[kind], "metrics": {}}
            for metric in METRICS:
                sm = seed_means(records, kind, region, metric, seeds)
                if len(sm) == 0:
                    continue
                lo, hi = bootstrap_ci(sm)
                reg[kind]["metrics"][metric] = {
                    "mean": float(sm.mean()),
                    "std": float(sm.std(ddof=1)) if len(sm) > 1 else 0.0,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "seed_means": sm.tolist(),
                }
        # comparisons on WEL and ISR
        reg["comparisons"] = {}
        for metric in ("WEL", "ISR"):
            reg["comparisons"][metric] = {}
            for kind in kinds:
                sm = seed_means(records, kind, region, metric, seeds)
                for ref in ("noop", "value_first", "hiercomm"):
                    if kind == ref or ref not in kinds:
                        continue
                    rm = seed_means(records, ref, region, metric, seeds)
                    if len(sm) > 1 and len(rm) > 1:
                        w = welch_ttest(sm, rm)
                        reg["comparisons"][metric][f"{kind}_vs_{ref}"] = {
                            "t": w["t_statistic"],
                            "p": w["p_value"],
                            "d": w["cohens_d"],
                        }
        summary["regions"][region] = reg
    return summary


def emit_latex(summary: dict, regions: list[str], kinds: list[str], out_path: Path) -> None:
    """Main results table (WEL/ISR with bootstrap CIs; bold best WEL per region)."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Main results} on the suppression-relevant regime "
        r"(5 seeds $\times$ episodes, seed means; brackets are bootstrap 95\% CIs). "
        r"Lower WEL / higher ISR is better; \textbf{bold} marks the best WEL per region.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Region & Policy & WEL $\downarrow$ & ISR $\uparrow$ \\",
        r"\midrule",
    ]
    for ri, region in enumerate(regions):
        reg = summary["regions"][region]
        present = [k for k in kinds if "WEL" in reg.get(k, {}).get("metrics", {})]
        best = min(present, key=lambda k: reg[k]["metrics"]["WEL"]["mean"]) if present else None
        label_region = {"saudi": "Saudi Arabia", "california": "California"}.get(region, region)
        lines.append(rf"\multirow{{{len(present)}}}{{*}}{{\textbf{{{label_region}}}}}")
        for k in present:
            wel = reg[k]["metrics"]["WEL"]
            isr = reg[k]["metrics"]["ISR"]
            w = f"{wel['mean']:.2f} \\small{{[{wel['ci_lo']:.1f}, {wel['ci_hi']:.1f}]}}"
            if k == best:
                w = f"\\textbf{{{wel['mean']:.2f}}} \\small{{[{wel['ci_lo']:.1f}, {wel['ci_hi']:.1f}]}}"
            lines.append(rf"& {LABELS[k]} & {w} & {isr['mean']:.3f} \\")
        lines.append(r"\midrule" if ri < len(regions) - 1 else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-3 train + evaluate + tables.")
    ap.add_argument("--regions", default="saudi,california")
    ap.add_argument("--methods", default=",".join(ALL_KINDS))
    ap.add_argument("--seeds", default="42,1042,2042,3042,4042")
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--train-steps", type=int, default=100000)
    ap.add_argument("--out", default="results/phase3")
    ap.add_argument("--quick", action="store_true", help="Tiny budget end-to-end validation")
    ap.add_argument("--force", action="store_true", help="Retrain even if checkpoint exists")
    ap.add_argument(
        "--eval-only", action="store_true", help="Skip training; evaluate existing checkpoints"
    )
    args = ap.parse_args()

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    kinds = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    episodes = 2 if args.quick else args.episodes
    steps = 1500 if args.quick else args.train_steps
    if args.quick:
        seeds = seeds[:2]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Phase-3 | regions={regions} methods={kinds} seeds={seeds} eps={episodes} "
        f"steps={steps} device={device}",
        flush=True,
    )

    # 1. Train learned methods (idempotent) --------------------------------------------------
    if not args.eval_only:
        for region in regions:
            for kind in kinds:
                if kind not in LEARNED:
                    continue
                path = checkpoint_path(kind, region, out_dir)
                if path.exists() and not args.force:
                    print(f"[SKIP] {kind}/{region} exists ({path.name})", flush=True)
                    continue
                train_method(kind, region, steps, args.quick, out_dir, device)

    # 2. Evaluate all methods ----------------------------------------------------------------
    records: list[dict] = []
    for region in regions:
        env = make_marl_env(region, "default", reward_cls=WELISRDeltaReward)
        for kind in kinds:
            if kind in LEARNED and not checkpoint_path(kind, region, out_dir).exists():
                print(f"[EVAL] skip {kind}/{region}: no checkpoint", flush=True)
                continue
            print(f"[EVAL] {kind}/{region}...", flush=True)
            records.extend(evaluate(env, kind, region, seeds, episodes, out_dir, device))
        env.close()

    # 3. Emit artifacts ----------------------------------------------------------------------
    raw = pd.DataFrame(records)
    raw.to_csv(out_dir / "phase3_raw.csv", index=False)
    present_kinds = [k for k in kinds if any(r["policy"] == k for r in records)]
    summary = build_summary(records, regions, present_kinds, seeds)
    (out_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2))
    emit_latex(summary, regions, present_kinds, out_dir / "phase3_main_table.tex")

    print("\n================ PHASE-3 SUMMARY ================", flush=True)
    for region in regions:
        print(f"\n{region.upper()}")
        reg = summary["regions"][region]
        for k in present_kinds:
            m = reg.get(k, {}).get("metrics", {})
            if "WEL" in m:
                print(
                    f"  {LABELS[k]:<22} WEL {m['WEL']['mean']:6.2f} "
                    f"[{m['WEL']['ci_lo']:5.2f},{m['WEL']['ci_hi']:5.2f}]  "
                    f"ISR {m['ISR']['mean']:.3f}  CE {m['CE']['mean']:.3f}"
                )
    print(f"\nArtifacts -> {out_dir}/ (phase3_raw.csv, phase3_summary.json, phase3_main_table.tex)")


if __name__ == "__main__":
    main()
