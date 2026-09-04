"""Generate the two raster figures for the paper from the *frozen* checkpoints and CSVs.

1. ``qualitative_comparison_california.png`` --- a final-frame 2x2 rollout comparison
   (HierComm vs CommNet, MAPPO, No-Op) on a representative California episode (seed 42),
   rendered at episode end so the outcome (assets defended vs lost) is visible. On-frame
   WEL/ISR are single-episode values; aggregates live in the main table.
2. ``learned_training_curves.png`` / ``.pdf`` --- WEL vs environment steps for the three
   learned methods (MAPPO, CommNet, HierComm), both regions, from the frozen train-curve CSVs.

Read-only over frozen artifacts; nothing is retrained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wildfire_marl.env.regimes import make_marl_env  # noqa: E402
from wildfire_marl.eval.metrics import (  # noqa: E402
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.eval.phase3_eval import (  # noqa: E402
    EVAL_OFFSET,
    LABELS,
    load_nets,
    select_actions,
)

CKPT = "results/wildfire_phase3_multiseed"
FIGDIR = Path("AAAI Template/Figures")
COMM = {"hiercomm_heur", "commnet"}
DEVICE = torch.device("cpu")
#: Paper-facing panel labels (the frozen LABELS tag hiercomm_heur as "heur. cmd").
PAPER_LABELS = {
    "hiercomm_heur": "HierComm (ours)",
    "commnet": "CommNet",
    "mappo": "Flat MARL (MAPPO)",
    "noop": "No-Op",
}


def _final_frame(renderer, env, kind, nets, region, seed):
    """Run one episode to completion; render only the final state."""
    obs, info = env.reset(seed=seed + EVAL_OFFSET)
    env.apply_target_compliance = False
    done, sc, ce, pos_hist = False, 0, 1.0, []
    while not done:
        acts = select_actions(env, kind, nets, obs, info, DEVICE, sc)
        obs, _r, term, trunc, info = env.step(acts)
        done = term["agent_0"] or trunc["agent_0"]
        ce = info[env.agents[0]].get("coordination_efficiency", ce)
        pos_hist.append(dict(env.agent_positions))
        sc += 1
    fs = env.env.fire_state
    wel = float(weighted_economic_loss(env.env.asset_type, fs, env.env.asset_values))
    isr = float(infrastructure_survival_rate(env.env.asset_type, fs))
    frame = renderer.render_frame(
        fire_state=fs.copy(),
        asset_type=env.env.asset_type,
        agent_positions=env.agent_positions.copy(),
        strategic_targets=env.strategic_targets.copy(),
        step_idx=sc - 1,
        wel=wel,
        isr=isr,
        ce=float(ce),
        region=region,
        policy_name=PAPER_LABELS.get(kind, LABELS.get(kind, kind)),
        prev_positions=pos_hist,
        comm_active=kind in COMM,
    )
    print(f"  {kind}/{region}: final WEL={wel:.1f} ISR={isr:.2f} (step {sc})", flush=True)
    return frame


def make_qualitative(region="california", seed=42):
    from wildfire_marl.viz.geo_renderer import GeoRenderer

    renderer = GeoRenderer(region, style="EsriWorldTopo", data_dir="data/cell2fire", theme="paper")
    env = make_marl_env(region, regime="default")
    policies = ["hiercomm_heur", "commnet", "mappo", "noop"]
    panels = []
    for kind in policies:
        nets = (
            load_nets(kind, region, CKPT, env.num_agents, DEVICE, seed=seed)
            if kind != "noop"
            else {}
        )
        renderer._fire_history.clear()
        panels.append(_final_frame(renderer, env, kind, nets, region, seed))
    env.close()

    w, h = panels[0].size
    pad = 8
    canvas = Image.new("RGBA", (2 * w + pad, 2 * h + pad), (255, 255, 255, 255))
    for j, img in enumerate(panels):
        r, c = divmod(j, 2)
        canvas.paste(img, (c * (w + pad), r * (h + pad)))
    scale = 300 / 150.0
    canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.LANCZOS)
    out = FIGDIR / f"qualitative_comparison_{region}.png"
    canvas.save(out, dpi=(300, 300))
    print(f"-> {out}", flush=True)


def make_training_curves():
    regions = ["saudi", "california"]
    methods = [
        ("mappo", "Flat MARL (MAPPO)", "#888888"),
        ("commnet", "CommNet", "#1f77b4"),
        ("hiercomm_heur", "HierComm (ours)", "#d62728"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, region in zip(axes, regions, strict=False):
        for stem, label, color in methods:
            f = Path(CKPT) / f"train_curve_checkpoint_{stem}_{region}_s42.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            wel = df["WEL"].rolling(25, min_periods=1).mean()
            ax.plot(df["env_steps"] / 1000.0, wel, label=label, color=color, lw=1.6)
        ax.set_title({"saudi": "Saudi Arabia", "california": "California"}[region], fontsize=10)
        ax.set_xlabel("Environment steps (thousands)", fontsize=9)
        ax.set_ylabel(r"WEL $\downarrow$ (rolling mean)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGDIR / f"learned_training_curves.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"-> {out}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("=== Qualitative comparison (final frame) ===", flush=True)
    make_qualitative("california", seed=42)
    print("=== Training curves ===", flush=True)
    make_training_curves()
    print("Done.", flush=True)
