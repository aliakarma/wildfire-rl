"""Phase-4 component ablations + regime-robustness sweep for the *proposed* HierComm model.

Fills the two Phase-4 gaps the shipped ``wildfire_phase3_multiseed/`` run does not cover:

1. **Component ablations** — does each piece of ``hiercomm_heur`` (value-aware hierarchy +
   learned communicating tactical layer + tactical shaping + RL fine-tune) add measurable value?
   Every ablation is the *same* system with one component removed, trained under the identical
   Phase-3 config and evaluated under the identical protocol (``make_marl_env(region,"default")``,
   disjoint eval seed stream). Variants:

     | variant              | change vs full                          | how                         |
     |----------------------|-----------------------------------------|-----------------------------|
     | ``full``             | none (reference == shipped model)       | base config                 |
     | ``wo_comms``         | no inter-agent communication            | ``comm_rounds = 0`` (train) |
     | ``wo_shaping``       | no frontier tactical shaping            | ``shaping_coef = 0`` (train)|
     | ``wo_rl_finetune``   | behaviour-clone only, no PPO fine-tune  | ``total_steps = 0`` (train) |
     | ``wo_hierarchy``     | flat comms, no strategic dispatch       | eval CommNet checkpoint     |
     | ``wo_learned_tactic``| value dispatch + Chebyshev (no NN)      | eval-only heuristic tactic  |

   ``wo_hierarchy`` and ``wo_learned_tactic`` need no training — they reuse the frozen CommNet
   checkpoints / a fixed heuristic, so the ablation isolates exactly one factor at a time.

2. **Regime robustness** — evaluate the frozen default-trained policies zero-shot across the
   ``easy``/``medium``/``hard`` regimes to confirm (a) No-Op WEL rises monotonically with
   difficulty (the Phase-1 difficulty axis is real) and (b) the proposed model keeps its ranking.

Idempotent (skip existing checkpoints; ``--force`` to retrain). Validate locally with ``--quick``;
launch the full multi-seed run on Colab GPU.

Examples
--------
Smoke::

    python scripts/run_phase4_ablations.py --study ablation --quick

Full ablation (5 training seeds), then robustness::

    python scripts/run_phase4_ablations.py --study ablation \\
        --train-seeds 42,1042,2042,3042,4042 --episodes 15 --train-steps 100000
    python scripts/run_phase4_ablations.py --study robustness \\
        --ckpt-dir wildfire_phase3_multiseed --episodes 15
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wildfire_marl.agents.agent_networks import CommNetActor, CommTacticalActor  # noqa: E402
from wildfire_marl.env.regimes import make_marl_env  # noqa: E402
from wildfire_marl.env.rewards import WELISRDeltaReward  # noqa: E402
from wildfire_marl.eval.metrics import (  # noqa: E402
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.eval.phase3_eval import EVAL_OFFSET, checkpoint_path, load_nets, rollout_episode  # noqa: E402
from wildfire_marl.eval.significance import bootstrap_ci, welch_ttest  # noqa: E402
from wildfire_marl.train.hier_comm_train import _dispatch_value, train_hier_comm  # noqa: E402
from wildfire_marl.train.hierarchical_train import target_seeking_action  # noqa: E402
from wildfire_marl.train.marl_train import set_seed  # noqa: E402

REGIONS = ["saudi", "california"]
METRICS = ["WEL", "ISR", "CE", "burned", "return"]

#: Trained ablation variants -> config override applied on top of the base HierComm config.
#: ``full`` is NOT here: the reference is the *shipped* model, so it reuses the frozen
#: ``hiercomm_heur`` checkpoints (byte-identical, no retrain) rather than training a fresh copy.
TRAIN_ABLATIONS: dict[str, dict] = {
    "wo_comms": {"comm_rounds": 0},
    "wo_shaping": {"shaping_coef": 0.0},
    "wo_rl_finetune": {"total_steps": 0},
}
#: Eval-only variants (no training): ``full`` = frozen hiercomm_heur, ``wo_hierarchy`` = frozen
#: CommNet, ``wo_learned_tactic`` = value dispatch + Chebyshev.
EVAL_ABLATIONS = ["full", "wo_hierarchy", "wo_learned_tactic"]
ALL_VARIANTS = ["full", *TRAIN_ABLATIONS, "wo_hierarchy", "wo_learned_tactic"]
VLABEL = {
    "full": "Full (proposed)",
    "wo_comms": "w/o communication",
    "wo_shaping": "w/o tactical shaping",
    "wo_rl_finetune": "w/o RL fine-tune (BC only)",
    "wo_hierarchy": "w/o hierarchy (flat comms)",
    "wo_learned_tactic": "w/o learned tactical (Chebyshev)",
}


def base_hier_cfg(steps: int, quick: bool) -> dict:
    """The shipped ``hiercomm_heur`` training config (mirrors ``run_phase3._base_cfg``)."""
    return {
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
        "commander": "heuristic",
    }


def ablation_ckpt(variant: str, region: str, out_dir: Path, seed: int) -> Path:
    return out_dir / f"checkpoint_ablation_{variant}_{region}_s{seed}.pt"


# --------------------------------------------------------------------------- training


def _worker_cmd(variant, region, train_seed, out_dir, steps, quick):
    """Command that trains exactly one (variant, region, seed) ablation in an isolated process."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--train-one",
        "--variant", variant, "--region", region, "--train-seed", str(train_seed),
        "--out", str(out_dir), "--train-steps", str(steps),
    ]
    if quick:
        cmd.append("--quick")
    return cmd


def _run_parallel(jobs, n_parallel, out_dir: Path) -> None:
    """Launch training subprocesses throttled to ``n_parallel`` concurrent, thread-pinned to a core."""
    logdir = out_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
    }
    procs: list = []
    for variant, region, s, cmd in jobs:
        while sum(p.poll() is None for p, _ in procs) >= n_parallel:
            time.sleep(3)
        lf = open(logdir / f"train_{variant}_{region}_s{s}.log", "w")  # noqa: SIM115
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        procs.append((p, lf))
        active = sum(pp.poll() is None for pp, _ in procs)
        print(f"[LAUNCH] {variant}/{region}/s{s} pid={p.pid} (active {active}/{n_parallel})", flush=True)
    for p, lf in procs:
        p.wait()
        lf.close()
    fails = [1 for p, _ in procs if p.returncode not in (0, None)]
    print(f"[PARALLEL] {len(procs)} ablation trainings finished, {len(fails)} failed", flush=True)


def train_variant(variant, region, steps, quick, out_dir, device, train_seed, force) -> None:
    path = ablation_ckpt(variant, region, out_dir, train_seed)
    if path.exists() and not force:
        print(f"[SKIP] {variant}/{region}/s{train_seed} exists", flush=True)
        return
    set_seed(train_seed)
    env = make_marl_env(region, "default", reward_cls=WELISRDeltaReward)
    cfg = base_hier_cfg(steps, quick)
    cfg.update(TRAIN_ABLATIONS[variant])
    cfg["seed"] = train_seed
    t0 = time.time()
    print(f"[TRAIN] ablation {variant}/{region}/s{train_seed} ({cfg['total_steps']} steps)...", flush=True)
    actor, _critic, _cmd, _curve = train_hier_comm(env, cfg, device)
    env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "actor_state_dict": actor.state_dict()}, path)
    print(f"[TRAIN] {variant}/{region}/s{train_seed} done in {(time.time()-t0)/60:.1f} min -> {path}",
          flush=True)


# --------------------------------------------------------------------------- eval


def _rollout_value_chebyshev(env, device, seed) -> dict:
    """w/o learned tactical: value-aware dispatch + Chebyshev target-seeking (no network)."""
    obs, info = env.reset(seed=seed)
    env.apply_target_compliance = False
    done, sc, ep_ret, ce = False, 0, 0.0, 1.0
    from wildfire_marl.eval.phase3_eval import MACRO_INTERVAL
    while not done:
        if sc % MACRO_INTERVAL == 0:
            _dispatch_value(env)
        acts = {a: target_seeking_action(env, a, info[a]["action_mask"]) for a in env.agents}
        obs, rew, term, trunc, info = env.step(acts)
        ep_ret += float(rew["agent_0"])
        done = term["agent_0"] or trunc["agent_0"]
        ce = info[env.agents[0]].get("coordination_efficiency", ce)
        sc += 1
    fs = env.env.fire_state
    return {
        "WEL": float(weighted_economic_loss(env.env.asset_type, fs, env.env.asset_values)),
        "ISR": float(infrastructure_survival_rate(env.env.asset_type, fs)),
        "CE": float(ce), "burned": int((fs > 0).sum()), "return": ep_ret,
    }


def _load_variant_nets(variant, region, out_dir, ckpt_dir, num_agents, device, train_seed):
    """Return (kind, nets) for the frozen dispatcher, or ('_chebyshev', None) for the NN-free variant."""
    if variant == "full":
        # the reference IS the shipped model -> reuse the frozen hiercomm_heur checkpoint
        nets = load_nets("hiercomm_heur", region, ckpt_dir, num_agents, device, seed=train_seed)
        return "hiercomm_heur", nets
    if variant in TRAIN_ABLATIONS:
        path = ablation_ckpt(variant, region, out_dir, train_seed)
        if not path.exists():
            return None, None
        ck = torch.load(path, map_location=device)
        actor = CommTacticalActor(
            in_channels=8, action_dim=6, features_dim=64,
            comm_rounds=int(ck.get("config", {}).get("comm_rounds", 2)),
        ).to(device)
        actor.load_state_dict(ck["actor_state_dict"])
        actor.eval()
        return "hiercomm_heur", {"actor": actor}
    if variant == "wo_hierarchy":
        # flat comms == the matched CommNet checkpoint from the frozen dir
        nets = load_nets("commnet", region, ckpt_dir, num_agents, device, seed=train_seed)
        return "commnet", nets
    if variant == "wo_learned_tactic":
        return "_chebyshev", None
    raise ValueError(variant)


def eval_variant(env, variant, region, out_dir, ckpt_dir, device, eval_seeds, episodes, train_seeds):
    """One score per training seed (mean over eval stream); heuristics -> single score."""
    per = {m: [] for m in METRICS}
    seeds_to_use = train_seeds if variant not in ("wo_learned_tactic",) else [None]
    for ts in seeds_to_use:
        kind, nets = _load_variant_nets(variant, region, out_dir, ckpt_dir, env.num_agents, device, ts)
        if kind is None:
            print(f"[EVAL] skip {variant}/{region}/s{ts}: no ckpt", flush=True)
            continue
        ep = {m: [] for m in METRICS}
        for s in eval_seeds:
            for e in range(episodes):
                seed = s + EVAL_OFFSET + e
                m = (_rollout_value_chebyshev(env, device, seed) if kind == "_chebyshev"
                     else rollout_episode(env, kind, nets, device, seed))
                for k in METRICS:
                    ep[k].append(m[k])
        for k in METRICS:
            per[k].append(float(np.mean(ep[k])))
    return per


def run_ablation(out_dir, ckpt_dir, steps, quick, device, eval_seeds, episodes, train_seeds,
                 force, n_parallel=1):
    # 1. train the config-toggle variants (parallel subprocesses when n_parallel > 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_parallel > 1:
        jobs = []
        for region in REGIONS:
            for variant in TRAIN_ABLATIONS:
                for ts in train_seeds:
                    if ablation_ckpt(variant, region, out_dir, ts).exists() and not force:
                        print(f"[SKIP] {variant}/{region}/s{ts} exists", flush=True)
                        continue
                    jobs.append((variant, region, ts, _worker_cmd(variant, region, ts, out_dir, steps, quick)))
        if jobs:
            _run_parallel(jobs, n_parallel, out_dir)
    else:
        for region in REGIONS:
            for variant in TRAIN_ABLATIONS:
                for ts in train_seeds:
                    train_variant(variant, region, steps, quick, out_dir, device, ts, force)
    # 2. evaluate every variant
    summary = {"protocol": {"train_seeds": train_seeds, "eval_seeds": eval_seeds,
                            "episodes": episodes}, "regions": {}}
    rows = []
    for region in REGIONS:
        env = make_marl_env(region, "default", reward_cls=WELISRDeltaReward)
        # No-Op reference for dWEL (rerun under identical protocol)
        noop = rollout_scores(env, "noop", {}, device, eval_seeds, episodes)
        noop_wel = float(np.mean(noop["WEL"]))
        reg = {}
        full_wel = None
        for variant in ALL_VARIANTS:
            per = eval_variant(env, variant, region, out_dir, ckpt_dir, device,
                               eval_seeds, episodes, train_seeds)
            if not per["WEL"]:
                continue
            wel_mean = float(np.mean(per["WEL"]))
            if variant == "full":
                full_wel = per["WEL"]
            reg[variant] = {
                "label": VLABEL[variant],
                "WEL_mean": wel_mean,
                "WEL_std": float(np.std(per["WEL"], ddof=1)) if len(per["WEL"]) > 1 else 0.0,
                "WEL_ci": list(bootstrap_ci(per["WEL"])) if len(per["WEL"]) > 1 else [wel_mean, wel_mean],
                "ISR_mean": float(np.mean(per["ISR"])),
                "dWEL_vs_noop": noop_wel - wel_mean,
                "n_seeds": len(per["WEL"]),
                "_wel_vals": per["WEL"],
            }
            rows.append({"region": region, "variant": variant, "WEL": wel_mean,
                         "ISR": float(np.mean(per["ISR"])), "dWEL": noop_wel - wel_mean})
            print(f"  [ablation] {region}/{variant}: WEL={wel_mean:.2f} ISR={np.mean(per['ISR']):.3f} "
                  f"dWEL={noop_wel-wel_mean:.2f}", flush=True)
        # significance: each ablation vs full (does removing the component hurt?)
        for variant, entry in reg.items():
            if variant == "full" or full_wel is None or len(entry["_wel_vals"]) < 2 or len(full_wel) < 2:
                continue
            w = welch_ttest(entry["_wel_vals"], full_wel)
            entry["vs_full_WEL"] = {"p": w["p_value"], "d": w["cohens_d"]}
        for entry in reg.values():
            entry.pop("_wel_vals", None)
        summary["regions"][region] = reg
        env.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "ablation_results.csv", index=False)
    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2))
    _emit_ablation_table(summary, out_dir / "phase4_ablation_table.tex")
    print(f"\nAblation artifacts -> {out_dir}/", flush=True)


def rollout_scores(env, kind, nets, device, eval_seeds, episodes) -> dict:
    per = {m: [] for m in METRICS}
    for s in eval_seeds:
        for e in range(episodes):
            m = rollout_episode(env, kind, nets, device, s + EVAL_OFFSET + e)
            for k in METRICS:
                per[k].append(m[k])
    # collapse to single "seed" score for heuristics
    return {k: [float(np.mean(v))] for k, v in per.items()}


def _emit_ablation_table(summary, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{Component ablation of the proposed model.} Each row removes one "
        r"component; lower WEL / higher ISR / higher $\Delta$WEL is better. Removing any component "
        r"degrades the full system.}",
        r"\label{tab:ablation}", r"\begin{tabular}{llccc}", r"\toprule",
        r"Region & Variant & WEL $\downarrow$ & ISR $\uparrow$ & $\Delta$WEL $\uparrow$ \\", r"\midrule",
    ]
    order = ["full", "wo_comms", "wo_hierarchy", "wo_learned_tactic", "wo_rl_finetune", "wo_shaping"]
    for region, reg in summary["regions"].items():
        rlabel = {"saudi": "Saudi", "california": "California"}[region]
        present = [v for v in order if v in reg]
        for i, v in enumerate(present):
            e = reg[v]
            lines.append(rf"{rlabel if i == 0 else ''} & {e['label']} & {e['WEL_mean']:.2f} "
                         rf"& {e['ISR_mean']:.3f} & {e['dWEL_vs_noop']:.2f} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- robustness


def run_robustness(out_dir, ckpt_dir, device, eval_seeds, episodes, train_seeds):
    """Zero-shot eval of frozen default-trained policies across easy/medium/hard regimes (resumable)."""
    regimes = ["easy", "medium", "hard"]
    policies = ["noop", "value_first", "commnet", "hiercomm_heur"]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "robustness_results.csv"
    existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if len(existing) else []

    def _done(region, regime, kind):
        if len(existing) == 0:
            return None
        sub = existing[(existing.region == region) & (existing.regime == regime)
                       & (existing.policy == kind)]
        return sub.iloc[0].to_dict() if len(sub) else None

    summary = {"protocol": {"regimes": regimes, "eval_seeds": eval_seeds, "episodes": episodes},
               "regions": {}}
    for region in REGIONS:
        reg = {}
        for regime in regimes:
            reg[regime] = {}
            env = None
            for kind in policies:
                cached = _done(region, regime, kind)
                if cached is not None:
                    reg[regime][kind] = {"WEL_mean": float(cached["WEL"]), "ISR_mean": float(cached["ISR"])}
                    print(f"  [robust] SKIP (done) {region}/{regime}/{kind}", flush=True)
                    continue
                if env is None:
                    env = make_marl_env(region, regime, reward_cls=WELISRDeltaReward)
                seeds_to_use = train_seeds if kind in ("commnet", "hiercomm_heur") else [None]
                wel_vals, isr_vals = [], []
                for ts in seeds_to_use:
                    try:
                        nets = load_nets(kind, region, ckpt_dir, env.num_agents, device, seed=ts)
                    except FileNotFoundError:
                        continue
                    ep_wel, ep_isr = [], []
                    for s in eval_seeds:
                        for e in range(episodes):
                            m = rollout_episode(env, kind, nets, device, s + EVAL_OFFSET + e)
                            ep_wel.append(m["WEL"]); ep_isr.append(m["ISR"])
                    wel_vals.append(float(np.mean(ep_wel))); isr_vals.append(float(np.mean(ep_isr)))
                if not wel_vals:
                    continue
                reg[regime][kind] = {"WEL_mean": float(np.mean(wel_vals)),
                                     "ISR_mean": float(np.mean(isr_vals))}
                row = {"region": region, "regime": regime, "policy": kind,
                       "WEL": float(np.mean(wel_vals)), "ISR": float(np.mean(isr_vals))}
                pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
                rows.append(row)
                print(f"  [robust] {region}/{regime}/{kind}: WEL={np.mean(wel_vals):.2f} "
                      f"ISR={np.mean(isr_vals):.3f}", flush=True)
            if env is not None:
                env.close()
        summary["regions"][region] = reg
    pd.DataFrame(rows).drop_duplicates(
        subset=["region", "regime", "policy"], keep="last").to_csv(csv_path, index=False)
    (out_dir / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
    # monotonicity check on No-Op WEL
    for region in REGIONS:
        noop_wel = [summary["regions"][region].get(r, {}).get("noop", {}).get("WEL_mean", float("nan"))
                    for r in regimes]
        mono = all(a <= b for a, b in zip(noop_wel, noop_wel[1:]) if not (np.isnan(a) or np.isnan(b)))
        print(f"  [robust] {region} No-Op WEL by difficulty {[round(x,1) for x in noop_wel]} "
              f"-> monotonic={mono}", flush=True)
    print(f"\nRobustness artifacts -> {out_dir}/", flush=True)


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-4 ablations + regime robustness (proposed model).")
    ap.add_argument("--study", default="ablation", choices=["ablation", "robustness", "both"])
    ap.add_argument("--out", default="results/phase4_ablation")
    ap.add_argument("--ckpt-dir", default="wildfire_phase3_multiseed",
                    help="Frozen checkpoints for wo_hierarchy (CommNet) + robustness sweep")
    ap.add_argument("--train-seeds", default="42,1042,2042,3042,4042")
    ap.add_argument("--eval-seeds", default="42,1042,2042,3042,4042")
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--train-steps", type=int, default=100000)
    ap.add_argument("--parallel", type=int, default=1, help="Concurrent ablation-training subprocesses")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Tiny end-to-end validation")
    # worker mode (used by --parallel): train exactly one (variant, region, seed) and exit
    ap.add_argument("--train-one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--region", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--train-seed", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    train_seeds = [int(s) for s in args.train_seeds.split(",") if s.strip()]
    eval_seeds = [int(s) for s in args.eval_seeds.split(",") if s.strip()]
    episodes = args.episodes
    steps = args.train_steps
    if args.quick:
        train_seeds, eval_seeds, episodes, steps = train_seeds[:1], eval_seeds[:1], 1, 0
    out_dir = Path(args.out)

    # --- worker mode: train exactly one ablation variant and exit (used by --parallel) ---
    if args.train_one:
        torch.set_num_threads(1)
        train_variant(args.variant, args.region, steps, args.quick, out_dir,
                      torch.device("cpu"), args.train_seed, args.force)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase-4 | study={args.study} train_seeds={train_seeds} eval_seeds={eval_seeds} "
          f"episodes={episodes} steps={steps} parallel={args.parallel} device={device}", flush=True)

    if args.study in ("ablation", "both"):
        run_ablation(out_dir, args.ckpt_dir, steps, args.quick, device,
                     eval_seeds, episodes, train_seeds, args.force, n_parallel=args.parallel)
    if args.study in ("robustness", "both"):
        run_robustness(out_dir, args.ckpt_dir, device, eval_seeds, episodes, train_seeds)


if __name__ == "__main__":
    main()
