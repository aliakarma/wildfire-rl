"""``wildfire-rl`` command-line interface.

Replaces notebook state with config-driven commands. Every subcommand accepts
``--config <yaml>`` and ``--set key=value`` dotlist overrides.

Examples
--------
    wildfire-rl info
    wildfire-rl build-tensors --region saudi_eastern_province --grid 32
    wildfire-rl train --config configs/experiment/multiseed.yaml
    wildfire-rl evaluate --set region.dir=california region.name=california
    wildfire-rl transfer --config configs/experiment/transfer.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wildfire_rl import __version__
from wildfire_rl.config import load_config, to_dict
from wildfire_rl.logging_utils import get_logger

logger = get_logger("wildfire_rl.cli")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    p.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=None,
        help="Dotlist overrides, e.g. ppo.total_timesteps=1000 seed=1",
    )


def _cfg(args) -> object:
    return load_config(args.config, args.overrides)


# --------------------------------------------------------------------- commands
def cmd_info(args) -> int:
    cfg = _cfg(args)
    print(json.dumps({"version": __version__, "config": to_dict(cfg)}, indent=2, default=str))
    return 0


def cmd_build_tensors(args) -> int:
    from wildfire_rl.data.tensor_stack import build_and_save
    from wildfire_rl.paths import data_dir

    grid_dir = data_dir() / args.region / "grids" / f"{args.grid}x{args.grid}"
    out = build_and_save(grid_dir)
    logger.info("Built state tensor -> %s", out)
    return 0


def cmd_train(args) -> int:
    import numpy as np

    from wildfire_rl.envs.base import make_env_factory
    from wildfire_rl.logging_utils import write_run_metadata
    from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir
    from wildfire_rl.train.ppo import train_ppo

    cfg = _cfg(args)
    tensor_path = region_tensor_path(cfg.region.dir, cfg.region.grid_size)
    tensor = np.load(tensor_path)
    factory = make_env_factory(state_tensor=tensor, config=cfg.env)

    ensure_dir(models_dir())
    for seed in cfg.seeds:
        run_id = f"train_{cfg.region.name}_seed_{seed}"
        run_dir = ensure_dir(results_dir() / "runs" / run_id)
        save_path = models_dir() / f"ppo_{cfg.region.name}_{cfg.region.grid_size}_seed_{seed}"

        # TensorBoard is optional (only if installed); the CSV curve is always written.
        tb_log = None
        try:
            import tensorboard  # noqa: F401

            tb_log = str(results_dir() / "runs" / "tb" / run_id)
        except ImportError:
            pass

        train_ppo(
            factory,
            cfg.ppo,
            seed=seed,
            save_path=save_path,
            tensorboard_log=tb_log,
            curve_path=run_dir / "curve.csv",
        )
        write_run_metadata(
            run_dir / "manifest.json",
            config_dict=to_dict(cfg),
            seed=seed,
            artifacts={"state_tensor": tensor_path, "model": f"{save_path}.zip"},
        )
    return 0


def cmd_evaluate(args) -> int:
    """Evaluate baselines (Random, NoOp) and optionally trained PPO models.

    AAAI-grade: outputs per-policy summary with 95% CIs, per-episode details,
    and significance tests (PPO vs each baseline).
    """
    import numpy as np
    import pandas as pd

    from wildfire_rl.envs.base import make_env_factory
    from wildfire_rl.eval.baselines import (
        FrontierPolicy,
        NearestFirePolicy,
        NoOpPolicy,
        RandomPolicy,
    )
    from wildfire_rl.eval.evaluate import evaluate_policy
    from wildfire_rl.eval.significance import (
        confidence_interval_95,
        format_significance,
        welch_ttest,
    )
    from wildfire_rl.paths import ensure_dir, models_dir, region_tensor_path, results_dir

    cfg = _cfg(args)
    tensor = np.load(region_tensor_path(cfg.region.dir, cfg.region.grid_size))
    factory = make_env_factory(state_tensor=tensor, config=cfg.env)

    # Baselines run without torch; the PPO model is optional.
    sample_env = factory()
    all_results: dict[str, dict] = {}

    # -- Baselines (incl. heuristic routers — the effective method) ----------
    gs = cfg.region.grid_size
    policies = {
        "random": RandomPolicy(sample_env.action_space, seed=cfg.seed),
        "noop": NoOpPolicy(sample_env.action_space),
        "nearest_fire": NearestFirePolicy(sample_env.action_space, grid_size=gs, env=sample_env),
        "frontier": FrontierPolicy(sample_env.action_space, grid_size=gs, env=sample_env),
    }

    from wildfire_rl.eval.loading import load_ppo_model

    # -- PPO models (auto-discover per-seed checkpoints) ---------------------
    for seed in cfg.seeds:
        model_path = models_dir() / f"ppo_{cfg.region.name}_{cfg.region.grid_size}_seed_{seed}.zip"
        if model_path.exists():
            policies[f"ppo_seed_{seed}"] = load_ppo_model(model_path, sample_env)
            logger.info("Loaded PPO seed %d <- %s", seed, model_path)

    # Fall back to explicit --model if no auto-discovered checkpoints.
    if args.model and not any(k.startswith("ppo_seed_") for k in policies):
        policies["ppo"] = load_ppo_model(Path(args.model), sample_env)
        logger.info("Loaded PPO <- %s", args.model)

    # -- Run evaluations -----------------------------------------------------
    for name, policy in policies.items():
        res = evaluate_policy(
            policy,
            factory,
            n_episodes=cfg.eval.n_episodes,
            base_seed=cfg.eval.base_seed,
            deterministic=cfg.eval.deterministic,
            scenario_seed_offset=cfg.eval.scenario_seed_offset,
            metrics_cfg=cfg.metrics,
        )
        all_results[name] = res
        logger.info("[%s] %s", name, res["summary"])

    # -- Build publication table with significance tests ---------------------
    rows = []
    ppo_episode_rewards: list[float] = []  # aggregate across seeds

    for name, res in all_results.items():
        rewards = [e["episode_reward"] for e in res["episodes"]]
        burned = [e["burned_cells"] for e in res["episodes"]]
        fire = [e["fire_intensity"] for e in res["episodes"]]

        ci_reward = confidence_interval_95(rewards)
        ci_burned = confidence_interval_95(burned)
        ci_fire = confidence_interval_95(fire)

        row = {
            "policy": name,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "reward_ci_lo": ci_reward[0],
            "reward_ci_hi": ci_reward[1],
            "burned_cells_mean": float(np.mean(burned)),
            "burned_cells_std": float(np.std(burned)),
            "burned_ci_lo": ci_burned[0],
            "burned_ci_hi": ci_burned[1],
            "fire_intensity_mean": float(np.mean(fire)),
            "fire_intensity_std": float(np.std(fire)),
            "fire_ci_lo": ci_fire[0],
            "fire_ci_hi": ci_fire[1],
            "n_episodes": len(rewards),
            "reward_mode": cfg.env.reward_mode,
        }

        if name.startswith("ppo"):
            ppo_episode_rewards.extend(rewards)

        rows.append(row)

    # Significance: PPO (aggregated) vs each baseline
    if ppo_episode_rewards:
        ppo_arr = np.array(ppo_episode_rewards)
        for baseline_name in ["random", "noop"]:
            if baseline_name in all_results:
                baseline_arr = np.array(
                    [e["episode_reward"] for e in all_results[baseline_name]["episodes"]]
                )
                test = welch_ttest(ppo_arr, baseline_arr)
                # Annotate the baseline row with the comparison
                for row in rows:
                    if row["policy"] == baseline_name:
                        row["p_vs_ppo"] = test["p_value"]
                        row["d_vs_ppo"] = test["cohens_d"]
                        row["sig_vs_ppo"] = format_significance(test["p_value"])
                        logger.info(
                            "  PPO vs %s: p=%.6f, d=%.2f (%s)",
                            baseline_name,
                            test["p_value"],
                            test["cohens_d"],
                            format_significance(test["p_value"]),
                        )

    # -- Write outputs -------------------------------------------------------
    out_dir = ensure_dir(results_dir())
    df = pd.DataFrame(rows)
    exp_name = getattr(cfg, "experiment_name", "multiseed")
    suffix = f"_{exp_name}" if exp_name != "multiseed" else ""
    csv_path = out_dir / f"eval_{cfg.region.name}{suffix}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    # Also write detailed JSON with per-episode data
    json_path = out_dir / f"eval_{cfg.region.name}{suffix}.json"
    json_data = {}
    for name, res in all_results.items():
        json_data[name] = {
            "summary": res["summary"],
            "episodes": res["episodes"],
        }
    json_path.write_text(json.dumps(json_data, indent=2))
    logger.info("Wrote %s", json_path)
    return 0


def cmd_transfer(args) -> int:
    from wildfire_rl.experiments.transfer_run import run_transfer

    out = run_transfer(
        args.config, args.overrides, seed=args.seed, allow_missing=args.allow_missing
    )
    logger.info("Transfer matrix written to %s", out)
    return 0


def cmd_make_figures(args) -> int:
    import numpy as np
    import pandas as pd

    from wildfire_rl.paths import ensure_dir, figures_dir, region_tensor_path, results_dir
    from wildfire_rl.viz.figures import (
        plot_ablation_bars,
        plot_baseline_comparison,
        plot_generalization_comparison,
        plot_marl_scaling,
        plot_state_tensor_comparison,
        plot_transfer_heatmap,
    )

    res, figs = results_dir(), ensure_dir(figures_dir())

    # Transfer heatmap — prefer normalized, fall back to the raw-mode matrix
    transfer_csv = res / "transfer_matrix.csv"
    if not transfer_csv.exists():
        transfer_csv = res / "transfer_matrix_raw.csv"
    if transfer_csv.exists():
        df = pd.read_csv(transfer_csv)
        plot_transfer_heatmap(
            df,
            "mean_reward",
            figs / "transfer_reward_heatmap.png",
            "Cross-region transfer (mean reward)",
        )
        logger.info("Wrote %s", figs / "transfer_reward_heatmap.png")
    else:
        logger.warning("No %s yet — run `wildfire-rl transfer` first.", transfer_csv)

    # Baseline comparison figures (per region) — accept the exact name or a config-suffixed variant
    # (e.g. eval_california_multiseed_california.csv), preferring the canonical name when present.
    for region in ["saudi", "california"]:
        eval_csv = res / f"eval_{region}.csv"
        if not eval_csv.exists():
            variants = sorted(res.glob(f"eval_{region}*.csv"))
            eval_csv = variants[0] if variants else eval_csv
        if eval_csv.exists():
            plot_baseline_comparison(
                eval_csv,
                figs / f"baseline_comparison_{region}.png",
                title=f"Policy Comparison — {region.title()}",
            )
            logger.info("Wrote baseline_comparison_%s.png (<- %s)", region, eval_csv.name)

    # Ablation bars
    ablation_csv = res / "ablation_results.csv"
    if ablation_csv.exists():
        plot_ablation_bars(ablation_csv, figs / "ablation_results.png")
        logger.info("Wrote ablation_results.png")

    # MARL scaling
    marl_csv = res / "marl_scaling_results.csv"
    if marl_csv.exists():
        plot_marl_scaling(marl_csv, figs / "marl_scaling.png")
        logger.info("Wrote marl_scaling.png")

    # State tensor comparison
    saudi_tensor_path = region_tensor_path("saudi_eastern_province", 32)
    ca_tensor_path = region_tensor_path("california", 32)
    if saudi_tensor_path.exists() and ca_tensor_path.exists():
        saudi_t = np.load(saudi_tensor_path)
        ca_t = np.load(ca_tensor_path)
        plot_state_tensor_comparison(saudi_t, ca_t, figs / "state_tensor_comparison.png")
        logger.info("Wrote state_tensor_comparison.png")

    # Generalization comparison
    fixed_csv = res / "eval_saudi.csv"
    gen_csv = res / "eval_saudi_generalization.csv"
    if fixed_csv.exists() and gen_csv.exists():
        plot_generalization_comparison(fixed_csv, gen_csv, figs / "generalization_results.png")
        logger.info("Wrote generalization_results.png")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wildfire-rl", description=__doc__)
    parser.add_argument("--version", action="version", version=f"wildfire-rl {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Print version + resolved config.")
    _add_config_args(p_info)
    p_info.set_defaults(func=cmd_info)

    p_bt = sub.add_parser("build-tensors", help="Stack channel layers into a state tensor.")
    p_bt.add_argument("--region", required=True, help="Region subdir under data/.")
    p_bt.add_argument("--grid", type=int, default=32)
    p_bt.set_defaults(func=cmd_build_tensors)

    p_tr = sub.add_parser("train", help="Train PPO (multi-seed) for a region.")
    _add_config_args(p_tr)
    p_tr.set_defaults(func=cmd_train)

    p_ev = sub.add_parser("evaluate", help="Evaluate baselines (+ optional PPO model).")
    _add_config_args(p_ev)
    p_ev.add_argument("--model", type=str, default=None, help="Path to a saved PPO .zip.")
    p_ev.set_defaults(func=cmd_evaluate)

    p_tf = sub.add_parser("transfer", help="Compute the full cross-region transfer matrix.")
    _add_config_args(p_tf)
    p_tf.add_argument("--seed", type=int, default=0, help="Which trained seed to load per region.")
    p_tf.add_argument(
        "--allow-missing",
        action="store_true",
        help="Dev dry-run: substitute RandomPolicy for missing models (NOT for reported results).",
    )
    p_tf.set_defaults(func=cmd_transfer)

    p_fig = sub.add_parser("make-figures", help="Regenerate paper figures from results.")
    p_fig.set_defaults(func=cmd_make_figures)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
