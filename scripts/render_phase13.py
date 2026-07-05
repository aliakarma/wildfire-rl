"""Phase 13 Smoke Test Orchestrator.
Runs lightweight rendering for rollouts, comparisons, compliance maps, and transfer heatmaps.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import random
import torch
import numpy as np
from PIL import Image

from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.agents.strategic_controller import StrategicController
from wildfire_marl.eval.metrics import burned_cells, weighted_economic_loss, infrastructure_survival_rate
from wildfire_marl.viz.rollout import render_rollout_frame, compile_rollout_gif
from wildfire_marl.viz.comparison import generate_side_by_side_comparison
from wildfire_marl.viz.strategic import plot_commander_decisions, plot_priority_heatmaps
from wildfire_marl.viz.transfer import plot_transfer_heatmap
from scripts.render_rollout import render_episode_rollout

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Run Phase 13 visualization smoke tests.")
    parser.add_argument("--output_dir", type=str, default="results/phase13")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=========================================")
    print("STARTING PHASE 13 SMOKE TESTS")
    print("=========================================\n")

    # Setup environment (Saudi)
    env = MultiAgentFireEnv(
        num_agents=3,
        crop_size=9,
        coordination_penalty=0.1,
        fire_map="Saudi",
        data_dir="data/cell2fire",
        max_steps=10,
        steps_per_action=60,
        observe_infra=True,
        catastrophe_weight=2.0,
        cascade_prob=0.1,
        infra_dir="data/cell2fire/Saudi",
    )

    # 1. Smoke Test 1: Render 1 short Saudi rollout GIF (10 steps)
    print("Smoke Test 1: Rendering short Saudi rollout GIF (10 steps)...")
    hier_ckpt = Path("results/runs/checkpoint_hierarchical_saudi.pt")
    commander = None
    if hier_ckpt.exists():
        ckpt = torch.load(hier_ckpt, map_location=device)
        commander = StrategicController(num_agents=env.num_agents).to(device)
        commander.load_state_dict(ckpt["commander_state_dict"])
        commander.eval()

    frames = render_episode_rollout(
        env, "Learned Hierarchical", commander, None, args.seed, device, max_steps=10
    )
    
    gif_path = out_dir / "smoke_rollout_saudi.gif"
    if frames:
        compile_rollout_gif(frames, gif_path, duration=300)
        print(f"  - Saved GIF: {gif_path}")
    else:
        print("  - Failed to render frames.")

    # 2. Smoke Test 2: Render 1 side-by-side comparison image
    print("\nSmoke Test 2: Generating side-by-side comparison image...")
    # Get frame 5 of Hierarchical
    h_frame = frames[5] if len(frames) > 5 else frames[0]
    
    # Get frame 5 of No-Op
    print("  - Rendering No-Op comparison frame...")
    noop_frames = render_episode_rollout(
        env, "No-Op", None, None, args.seed, device, max_steps=6
    )
    n_frame = noop_frames[5] if len(noop_frames) > 5 else noop_frames[0]
    
    comparison_img_path = out_dir / "smoke_comparison_noop_vs_hier.png"
    generate_side_by_side_comparison(
        n_frame, h_frame, "No-Op Baseline (Left)", "Learned Hierarchical (Right)",
        comparison_img_path
    )
    print(f"  - Saved comparison image: {comparison_img_path}")

    # 3. Smoke Test 3: Generate compliance heatmap & commander decisions
    print("\nSmoke Test 3: Generating strategic decisions plot...")
    decision_plot_path = out_dir / "smoke_commander_decisions.png"
    plot_commander_decisions(
        fire_state=env.env.fire_state,
        criticality=env.env.criticality,
        agent_positions=env.agent_positions,
        strategic_targets=env.strategic_targets,
        save_path=decision_plot_path
    )
    print(f"  - Saved strategic decision plot: {decision_plot_path}")

    # 4. Smoke Test 4: Generate transfer heatmap
    print("\nSmoke Test 4: Generating transfer heatmap...")
    # Load 2x2 transfer matrix dummy values or existing values
    matrix = np.array([
        [1.00, 0.95],
        [0.98, 1.00]
    ])
    transfer_heatmap_path = out_dir / "smoke_transfer_heatmap.png"
    plot_transfer_heatmap(
        matrix_2x2=matrix,
        row_labels=["Saudi Policy", "California Policy"],
        col_labels=["Saudi Env", "California Env"],
        title="Transfer Robustness Scores (TRS)",
        save_path=transfer_heatmap_path
    )
    print(f"  - Saved transfer heatmap: {transfer_heatmap_path}")

    env.close()

    # 5. Smoke Test 5: Verify no crashes & output files
    print("\nSmoke Test 5: Verifying output files exist...")
    all_exist = True
    for p in [gif_path, comparison_img_path, decision_plot_path, transfer_heatmap_path]:
        if p.exists():
            print(f"  - Verified: {p} (size: {p.stat().st_size} bytes)")
        else:
            print(f"  - MISSING: {p}")
            all_exist = False

    if all_exist:
        print("\n=========================================")
        print("ALL SMOKE TESTS PASSED SUCCESSFULLY!")
        print("=========================================")
    else:
        print("\n=========================================")
        print("SMOKE TESTS FAILED!")
        print("=========================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
