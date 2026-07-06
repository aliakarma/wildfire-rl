"""Transfer rendering script to compile cross-region transfer rollouts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import random

import torch
from scripts.render_rollout import render_episode_rollout

from wildfire_marl.agents.strategic_controller import StrategicController
from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.viz.rollout import compile_rollout_gif


def set_seed(seed: int):
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Render cross-region transfer rollouts.")
    parser.add_argument("--source_region", type=str, required=True, choices=["saudi", "california"])
    parser.add_argument("--target_region", type=str, required=True, choices=["saudi", "california"])
    parser.add_argument("--output_gif", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    map_name = "Saudi" if args.target_region.lower() == "saudi" else "California"
    data_dir = "data/cell2fire"
    infra_dir = f"{data_dir}/{map_name}"

    env = MultiAgentFireEnv(
        num_agents=3,
        crop_size=9,
        coordination_penalty=0.1,
        fire_map=map_name,
        data_dir=data_dir,
        max_steps=150,
        steps_per_action=60,
        observe_infra=True,
        catastrophe_weight=2.0,
        cascade_prob=0.1,
        infra_dir=infra_dir,
    )

    # Load policy from source training region
    hierarchical_ckpt = Path(f"results/runs/checkpoint_hierarchical_{args.source_region}.pt")
    if not hierarchical_ckpt.exists():
        raise FileNotFoundError(f"Missing source policy checkpoint: {hierarchical_ckpt}")

    ckpt = torch.load(hierarchical_ckpt, map_location=device)
    commander = StrategicController(num_agents=env.num_agents).to(device)
    commander.load_state_dict(ckpt["commander_state_dict"])
    commander.eval()

    print(
        f"Rendering transfer rollout from {args.source_region} training to {args.target_region} map ({args.steps} steps)..."
    )

    frames = render_episode_rollout(
        env, "Learned Hierarchical", commander, None, args.seed, device, max_steps=args.steps
    )
    env.close()

    if frames:
        compile_rollout_gif(frames, args.output_gif)
        print(f"Saved transfer rollout animation to: {args.output_gif}")
    else:
        print("Error: No frames rendered.")


if __name__ == "__main__":
    main()
