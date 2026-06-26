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

from wildfire_rl import __version__
from wildfire_rl.config import load_config, to_dict
from wildfire_rl.logging_utils import get_logger

logger = get_logger("wildfire_rl.cli")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    p.add_argument(
        "--set", dest="overrides", nargs="*", default=None,
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
        save_path = models_dir() / f"ppo_{cfg.region.name}_{cfg.region.grid_size}_seed_{seed}"
        train_ppo(factory, cfg.ppo, seed=seed, save_path=save_path)
        write_run_metadata(
            results_dir() / "runs" / f"train_{cfg.region.name}_seed_{seed}.json",
            config_dict=to_dict(cfg),
            seed=seed,
        )
    return 0


def cmd_evaluate(args) -> int:
    import numpy as np

    from wildfire_rl.envs.base import make_env_factory
    from wildfire_rl.eval.baselines import NoOpPolicy, RandomPolicy
    from wildfire_rl.eval.evaluate import evaluate_policy
    from wildfire_rl.paths import ensure_dir, region_tensor_path, results_dir

    cfg = _cfg(args)
    tensor = np.load(region_tensor_path(cfg.region.dir, cfg.region.grid_size))
    factory = make_env_factory(state_tensor=tensor, config=cfg.env)

    # Baselines run without torch; the PPO model is optional (loaded if --model given).
    sample_env = factory()
    results = {}
    policies = {
        "random": RandomPolicy(sample_env.action_space, seed=cfg.seed),
        "noop": NoOpPolicy(sample_env.action_space),
    }
    if args.model:
        from stable_baselines3 import PPO

        policies["ppo"] = PPO.load(args.model)

    for name, policy in policies.items():
        res = evaluate_policy(
            policy, factory, n_episodes=cfg.eval.n_episodes,
            base_seed=cfg.eval.base_seed, deterministic=cfg.eval.deterministic,
            metrics_cfg=cfg.metrics,
        )
        results[name] = res["summary"]
        logger.info("[%s] %s", name, res["summary"])

    out = ensure_dir(results_dir()) / f"eval_{cfg.region.name}.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out)
    return 0


def cmd_transfer(args) -> int:
    from wildfire_rl.experiments.transfer_run import run_transfer

    out = run_transfer(args.config, args.overrides, seed=args.seed)
    logger.info("Transfer matrix written to %s", out)
    return 0


def cmd_make_figures(args) -> int:
    import pandas as pd

    from wildfire_rl.paths import ensure_dir, figures_dir, results_dir
    from wildfire_rl.viz.figures import plot_transfer_heatmap

    res, figs = results_dir(), ensure_dir(figures_dir())
    transfer_csv = res / "transfer_matrix.csv"
    if transfer_csv.exists():
        df = pd.read_csv(transfer_csv)
        plot_transfer_heatmap(df, "mean_reward", figs / "transfer_reward_heatmap.png",
                              "Cross-region transfer (mean reward)")
        logger.info("Wrote %s", figs / "transfer_reward_heatmap.png")
    else:
        logger.warning("No %s yet — run `wildfire-rl transfer` first.", transfer_csv)
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
    p_tf.set_defaults(func=cmd_transfer)

    p_fig = sub.add_parser("make-figures", help="Regenerate paper figures from results.")
    p_fig.set_defaults(func=cmd_make_figures)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
