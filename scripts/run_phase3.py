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
import os
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
    if kind in ("mappo", "commnet"):
        cfg = {
            "total_steps": steps,
            "n_steps": 1024,
            "batch_size": 64,
            "ppo_epochs": 4,
            "clip_eps": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "comm_rounds": 2,
        }
    elif kind == "qmix":
        # Documented QMIX retrain: the original Phase-3 config (lr 5e-4, batch 32,
        # target-sync 200, 60% eps-decay, 5k buffer) flatlined at the No-Op level. This tuned
        # config gives the value-factorised learner a fairer chance on the sparse WEL/ISR
        # reward: higher LR, larger batch + replay buffer, more frequent target sync, and a
        # longer exploration schedule.
        cfg = {
            "total_steps": steps,
            "batch_size": 64,
            "gamma": 0.99,
            "lr": 1e-3,
            "target_update_interval": 100,
            "epsilon_end": 0.05,
            "epsilon_decay_frac": 0.7,
            "buffer_capacity": 20000,
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


def train_method(
    kind: str,
    region: str,
    steps: int,
    quick: bool,
    out_dir: Path,
    device,
    train_seed: int = 42,
    suffix_seed: bool = False,
) -> None:
    """Train one learned method with a fixed ``train_seed`` and save its checkpoint.

    ``train_seed`` seeds both the networks (``set_seed``) and — via ``cfg['seed']`` — the
    deterministic env-reset stream, so the whole run is reproducible. ``suffix_seed`` names the
    checkpoint ``..._s<seed>.pt`` for the multi-training-seed sweep.
    """
    set_seed(train_seed)
    env = make_marl_env(region, "default", reward_cls=WELISRDeltaReward)
    cfg = _base_cfg(kind, steps, quick)
    cfg["seed"] = train_seed
    t0 = time.time()
    print(f"\n[TRAIN] {kind} / {region} / seed {train_seed} ({steps} steps)...", flush=True)
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
    path = checkpoint_path(kind, region, out_dir, seed=train_seed if suffix_seed else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    if curve:
        pd.DataFrame(curve).to_csv(out_dir / f"train_curve_{path.stem}.csv", index=False)
    print(
        f"[TRAIN] {kind}/{region}/s{train_seed} done in {(time.time() - t0) / 60:.1f} min -> {path}",
        flush=True,
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


# --------------------------------------------------------------------------- multi-seed sweep


def _worker_cmd(method: str, region: str, train_seed: int, out_dir: Path, steps: int, quick: bool):
    """Command that trains exactly one (method, region, seed) job in an isolated process."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--train-one",
        "--method",
        method,
        "--region",
        region,
        "--train-seed",
        str(train_seed),
        "--out",
        str(out_dir),
        "--train-steps",
        str(steps),
    ]
    if quick:
        cmd.append("--quick")
    return cmd


def _run_parallel(jobs: list, n_parallel: int, out_dir: Path) -> None:
    """Launch training subprocesses, throttled to ``n_parallel`` concurrent and thread-pinned."""
    import subprocess

    logdir = out_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    # Single-thread + CPU per worker so N jobs map to N cores (no oversubscription).
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    procs: list = []
    for m, r, s, cmd in jobs:
        while sum(p.poll() is None for p, _ in procs) >= n_parallel:
            time.sleep(3)
        lf = open(logdir / f"train_{m}_{r}_s{s}.log", "w")  # noqa: SIM115 (closed after p.wait())
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        procs.append((p, lf))
        active = sum(pp.poll() is None for pp, _ in procs)
        print(f"[LAUNCH] {m}/{r}/s{s} pid={p.pid} (active {active}/{n_parallel})", flush=True)
    for p, lf in procs:
        p.wait()
        lf.close()
    fails = [1 for p, _ in procs if p.returncode not in (0, None)]
    print(f"[PARALLEL] {len(procs)} jobs finished, {len(fails)} failed", flush=True)


def _score(records: list, metric: str) -> float:
    return float(np.mean([r[metric] for r in records])) if records else float("nan")


def build_multiseed_summary(
    scores: dict, regions: list, kinds: list, train_seeds: list, proposed: str = "hiercomm_heur"
) -> dict:
    """Summary with mean +/- std OVER TRAINING SEEDS and proposed-vs-baseline tests.

    Learned methods contribute one score per training seed (n = #seeds); heuristics contribute a
    single deterministic score. Proposed-vs-learned uses Welch; proposed-vs-heuristic uses a
    one-sample t-test against the heuristic's fixed value.
    """
    from scipy import stats

    summary: dict = {
        "protocol": {"train_seeds": train_seeds, "unit": "per-training-seed eval scores"},
        "regions": {},
    }
    for region in regions:
        reg: dict = {}
        for kind in kinds:
            if (kind, region) not in scores:
                continue
            reg[kind] = {"label": LABELS[kind], "metrics": {}}
            for metric in METRICS:
                vals = np.asarray(scores[(kind, region)].get(metric, []), dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                mean = float(vals.mean())
                std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                lo, hi = bootstrap_ci(vals) if len(vals) > 1 else (mean, mean)
                reg[kind]["metrics"][metric] = {
                    "mean": mean,
                    "std": std,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_train_seeds": int(len(vals)),
                    "values": vals.tolist(),
                }
        reg["comparisons"] = {}
        if (proposed, region) in scores:
            for metric in ("WEL", "ISR"):
                reg["comparisons"][metric] = {}
                pv = np.asarray(scores[(proposed, region)][metric], dtype=float)
                pv = pv[~np.isnan(pv)]
                for kind in kinds:
                    if kind == proposed or (kind, region) not in scores:
                        continue
                    ov = np.asarray(scores[(kind, region)][metric], dtype=float)
                    ov = ov[~np.isnan(ov)]
                    if len(pv) < 2 or len(ov) == 0:
                        continue
                    if len(ov) > 1:
                        t, p = stats.ttest_ind(pv, ov, equal_var=False)
                        pooled = np.sqrt((pv.var(ddof=1) + ov.var(ddof=1)) / 2)
                        d = (pv.mean() - ov.mean()) / pooled if pooled > 0 else 0.0
                        test = "welch"
                    else:
                        t, p = stats.ttest_1samp(pv, ov[0])
                        sd = pv.std(ddof=1)
                        d = (pv.mean() - ov[0]) / sd if sd > 0 else 0.0
                        test = "one-sample"
                    reg["comparisons"][metric][f"proposed_vs_{kind}"] = {
                        "t": float(t),
                        "p": float(p),
                        "d": float(d),
                        "test": test,
                    }
        summary["regions"][region] = reg
    return summary


def emit_latex_multiseed(
    summary: dict, regions: list, kinds: list, out_path: Path, n_seeds: int
) -> None:
    """Main table with WEL/ISR as mean +/- std over training seeds; bold best WEL per region."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Main results} (mean $\pm$ std over "
        + str(n_seeds)
        + r" training seeds; heuristics are deterministic). Lower WEL / higher ISR is better; "
        r"\textbf{bold} marks the best WEL per region.}",
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
            wel, isr = reg[k]["metrics"]["WEL"], reg[k]["metrics"]["ISR"]
            wcell = f"{wel['mean']:.2f} $\\pm$ {wel['std']:.2f}"
            if k == best:
                wcell = f"\\textbf{{{wel['mean']:.2f}}} $\\pm$ {wel['std']:.2f}"
            lines.append(rf"& {LABELS[k]} & {wcell} & {isr['mean']:.3f} $\pm$ {isr['std']:.3f} \\")
        lines.append(r"\midrule" if ri < len(regions) - 1 else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines))


def run_multiseed(
    kinds,
    regions,
    train_seeds,
    eval_seeds,
    episodes,
    steps,
    quick,
    out_dir: Path,
    n_parallel: int,
    force: bool,
) -> None:
    """Train each learned method over several TRAINING seeds (parallel), evaluate each seed's
    checkpoint on the fixed eval stream, and report mean +/- std over training seeds."""
    learned = [k for k in kinds if k in LEARNED]
    heur = [k for k in kinds if k not in LEARNED]

    jobs = []
    for r in regions:
        for m in learned:
            for s in train_seeds:
                if checkpoint_path(m, r, out_dir, seed=s).exists() and not force:
                    print(f"[SKIP] {m}/{r}/s{s} exists", flush=True)
                    continue
                jobs.append((m, r, s, _worker_cmd(m, r, s, out_dir, steps, quick)))
    if jobs:
        if n_parallel > 1:
            _run_parallel(jobs, n_parallel, out_dir)
        else:
            for m, r, s, _cmd in jobs:
                train_method(
                    m, r, steps, quick, out_dir, torch.device("cpu"), train_seed=s, suffix_seed=True
                )

    device = torch.device("cpu")
    scores: dict = {}
    raw: list = []
    for r in regions:
        env = make_marl_env(r, "default", reward_cls=WELISRDeltaReward)
        for m in learned:
            per = {metric: [] for metric in METRICS}
            for s in train_seeds:
                if not checkpoint_path(m, r, out_dir, seed=s).exists():
                    print(f"[EVAL] skip {m}/{r}/s{s}: no ckpt", flush=True)
                    continue
                recs = evaluate(env, m, r, eval_seeds, episodes, out_dir, device, train_seed=s)
                for rec in recs:
                    rec["train_seed"] = s
                raw.extend(recs)
                for metric in METRICS:
                    per[metric].append(_score(recs, metric))
            scores[(m, r)] = per
            print(f"[EVAL] {m}/{r}: WEL per-seed={[round(x, 2) for x in per['WEL']]}", flush=True)
        for h in heur:
            recs = evaluate(env, h, r, eval_seeds, episodes, out_dir, device)
            for rec in recs:
                rec["train_seed"] = None
            raw.extend(recs)
            scores[(h, r)] = {metric: [_score(recs, metric)] for metric in METRICS}
        env.close()

    present = [k for k in kinds if any(k2 == k for (k2, _r) in scores)]
    summary = build_multiseed_summary(scores, regions, present, train_seeds)
    pd.DataFrame(raw).to_csv(out_dir / "phase3_raw.csv", index=False)
    (out_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2))
    emit_latex_multiseed(
        summary, regions, present, out_dir / "phase3_main_table.tex", len(train_seeds)
    )

    print("\n========= MULTI-SEED SUMMARY (mean +/- std over training seeds) =========")
    for region in regions:
        print(f"\n{region.upper()}")
        reg = summary["regions"][region]
        for k in present:
            m = reg.get(k, {}).get("metrics", {})
            if "WEL" in m:
                print(
                    f"  {LABELS[k]:<22} WEL {m['WEL']['mean']:6.2f} +/- {m['WEL']['std']:.2f}  "
                    f"ISR {m['ISR']['mean']:.3f}  (n={m['WEL']['n_train_seeds']})"
                )
        for k, c in reg.get("comparisons", {}).get("WEL", {}).items():
            print(f"    {k}: p={c['p']:.4f} d={c['d']:.2f} [{c['test']}]")
    print(f"\nArtifacts -> {out_dir}/ (phase3_raw.csv, phase3_summary.json, phase3_main_table.tex)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-3 train + evaluate + tables.")
    ap.add_argument("--regions", default="saudi,california")
    ap.add_argument("--methods", default=",".join(ALL_KINDS))
    ap.add_argument("--seeds", default="42,1042,2042,3042,4042", help="Eval seed-groups (held-out)")
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--train-steps", type=int, default=100000)
    ap.add_argument("--out", default="results/phase3")
    ap.add_argument("--quick", action="store_true", help="Tiny budget end-to-end validation")
    ap.add_argument("--force", action="store_true", help="Retrain even if checkpoint exists")
    ap.add_argument(
        "--eval-only", action="store_true", help="Skip training; evaluate existing checkpoints"
    )
    ap.add_argument(
        "--train-seeds",
        default=None,
        help="Comma TRAINING seeds -> multi-seed protocol (mean +/- std over training seeds)",
    )
    ap.add_argument("--parallel", type=int, default=1, help="Concurrent training subprocesses")
    ap.add_argument("--train-one", action="store_true", help=argparse.SUPPRESS)  # worker mode
    ap.add_argument("--method", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--region", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--train-seed", type=int, default=None, help=argparse.SUPPRESS)
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

    # --- worker mode: train exactly one (method, region, seed) and exit (used by --parallel) ---
    if args.train_one:
        torch.set_num_threads(1)
        train_method(
            args.method,
            args.region,
            steps,
            args.quick,
            out_dir,
            torch.device("cpu"),
            train_seed=args.train_seed,
            suffix_seed=True,
        )
        return

    # --- multi-seed protocol: train N training seeds, aggregate mean +/- std over them ---------
    if args.train_seeds:
        train_seeds = [int(s) for s in args.train_seeds.split(",") if s.strip()]
        if args.quick:
            train_seeds = train_seeds[:2]
        print(
            f"Phase-3 MULTI-SEED | regions={regions} methods={kinds} train_seeds={train_seeds} "
            f"eval_seeds={seeds} parallel={args.parallel} steps={steps}",
            flush=True,
        )
        run_multiseed(
            kinds,
            regions,
            train_seeds,
            seeds,
            episodes,
            steps,
            args.quick,
            out_dir,
            args.parallel,
            args.force,
        )
        return

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
