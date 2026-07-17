"""Phase-6 transfer GIFs: visualize how a policy trained on region A behaves on region B.

For the proposed model (and CommNet as a baseline), renders the *native* rollout (A trained,
A evaluated) next to the *transferred* rollout (A trained, B evaluated) on the same eval region,
so the reader can see whether coordinated asset defense survives a domain shift. Reuses the
Phase-5 GIS renderer (``viz.geo_renderer.GeoRenderer``) and the frozen action dispatcher
(``phase3_eval.select_actions``) — behaviour is identical to the evaluation harness.

Read-only over the frozen checkpoints; nothing in the RL/env stack is modified.

Example::

    python scripts/render_phase6_transfer.py --ckpt-dir wildfire_phase3_multiseed \\
        --out figures/phase6 --policies hiercomm_heur,commnet --max-steps 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wildfire_marl.env.regimes import make_marl_env  # noqa: E402
from wildfire_marl.eval.metrics import (  # noqa: E402
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.eval.phase3_eval import EVAL_OFFSET, LABELS, load_nets, select_actions  # noqa: E402
from wildfire_marl.viz.rollout import compile_rollout_gif  # noqa: E402

REGIONS = ["saudi", "california"]
OTHER = {"saudi": "california", "california": "saudi"}
REGION_ABBR = {"saudi": "SA", "california": "CA"}
COMM_POLICIES = {"hiercomm_heur", "hiercomm", "commnet"}


def rollout_frames(renderer, env, kind, nets, device, seed, eval_region, label, max_steps):
    """One episode on ``eval_region``; returns (frames, interest_scores) with a custom title."""
    from wildfire_marl.viz.geo_renderer import frame_interest_score

    obs, info = env.reset(seed=seed + EVAL_OFFSET)
    env.apply_target_compliance = False
    done, sc, ce = False, 0, 1.0
    frames, scores, pos_hist = [], [], []
    comm_active = kind in COMM_POLICIES
    while not done and sc < max_steps:
        acts = select_actions(env, kind, nets, obs, info, device, sc)
        obs, _rew, term, trunc, info = env.step(acts)
        done = term["agent_0"] or trunc["agent_0"]
        ce = info[env.agents[0]].get("coordination_efficiency", ce)
        fs = env.env.fire_state
        wel = weighted_economic_loss(env.env.asset_type, fs, env.env.asset_values)
        isr = infrastructure_survival_rate(env.env.asset_type, fs)
        pos_hist.append(dict(env.agent_positions))
        frames.append(renderer.render_frame(
            fire_state=fs.copy(), asset_type=env.env.asset_type,
            agent_positions=env.agent_positions.copy(),
            strategic_targets=env.strategic_targets.copy(), step_idx=sc,
            wel=float(wel), isr=float(isr), ce=float(ce), region=eval_region,
            policy_name=label, prev_positions=pos_hist, comm_active=comm_active,
        ))
        scores.append(frame_interest_score(fs, env.env.asset_type))
        sc += 1
    return frames, scores


def _pair_snapshot(native, transfer, path, dpi=300):
    """Compose a 2-panel native | transferred PNG at each side's most informative frame."""
    imgs = [native, transfer]
    w, h = imgs[0].size
    pad = 8
    canvas = Image.new("RGBA", (2 * w + pad, h), (26, 26, 46, 255))
    canvas.paste(native, (0, 0))
    canvas.paste(transfer, (w + pad, 0))
    scale = dpi / 150.0
    canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.LANCZOS)
    canvas.save(path, dpi=(dpi, dpi))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-6 transfer GIF renderer (frozen checkpoints).")
    ap.add_argument("--ckpt-dir", default="wildfire_phase3_multiseed")
    ap.add_argument("--out", default="figures/phase6")
    ap.add_argument("--policies", default="hiercomm_heur,commnet")
    ap.add_argument("--regions", default="saudi,california")
    ap.add_argument("--seed", type=int, default=42, help="Training-seed checkpoint + eval seed group")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--style", default="EsriWorldTopo")
    ap.add_argument("--theme", default="paper", choices=["paper", "presentation"])
    ap.add_argument("--data-dir", default="data/cell2fire")
    ap.add_argument("--duration", type=int, default=200)
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    from wildfire_marl.viz.geo_renderer import GeoRenderer

    for eval_region in regions:
        train_region = OTHER[eval_region]
        renderer = GeoRenderer(eval_region, style=args.style, data_dir=args.data_dir, theme=args.theme)
        env = make_marl_env(eval_region, regime="default")
        print(f"\n=== eval on {eval_region.upper()} (transfer from {train_region.upper()}) ===",
              flush=True)
        for kind in policies:
            base = LABELS.get(kind, kind)
            # native: trained & evaluated on eval_region
            native_nets = load_nets(kind, eval_region, args.ckpt_dir, env.num_agents, device,
                                    seed=args.seed)
            renderer._fire_history.clear()
            nat_frames, nat_scores = rollout_frames(
                renderer, env, kind, native_nets, device, args.seed, eval_region,
                f"{base}  [native {REGION_ABBR[eval_region]}]", args.max_steps)
            # transferred: trained on the other region, evaluated here
            xfer_nets = load_nets(kind, train_region, args.ckpt_dir, env.num_agents, device,
                                  seed=args.seed)
            renderer._fire_history.clear()
            xfer_frames, xfer_scores = rollout_frames(
                renderer, env, kind, xfer_nets, device, args.seed, eval_region,
                f"{base}  [{REGION_ABBR[train_region]}→{REGION_ABBR[eval_region]}]",
                args.max_steps)

            compile_rollout_gif(nat_frames, out_dir / f"rollout_{kind}_{eval_region}_native.gif",
                                duration=args.duration)
            compile_rollout_gif(xfer_frames,
                                out_dir / f"rollout_{kind}_{train_region}_to_{eval_region}.gif",
                                duration=args.duration)
            ni = int(np.argmax(nat_scores)) if nat_scores else 0
            xi = int(np.argmax(xfer_scores)) if xfer_scores else 0
            _pair_snapshot(nat_frames[ni], xfer_frames[xi],
                           out_dir / f"fig_transfer_{kind}_{eval_region}.png")
            print(f"  {kind}: native({len(nat_frames)}f) + transfer({len(xfer_frames)}f) "
                  f"-> {eval_region}", flush=True)
        env.close()

    print(f"\nPhase-6 transfer GIFs -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
