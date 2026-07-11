"""Phase-5 rollout GIF renderer with GIS basemap.

Generates per-policy rollout GIFs, a 2x2 comparison grid, publication-quality PNGs
(300 DPI), and optionally MP4 video / PDF vector, using the *frozen* seed-42
checkpoints from ``wildfire_phase3_multiseed/``.

Policy behaviour is identical to the evaluation pipeline — it calls
``phase3_eval.select_actions`` — so the GIFs are faithful to the frozen table.

Generation commands
-------------------
::

    # === AAAI paper figures (default paper theme) ===
    python scripts/render_phase5_gifs.py \\
        --ckpt-dir wildfire_phase3_multiseed --out figures/phase5 \\
        --seed 42 --regions saudi,california --max-steps 150 \\
        --style EsriWorldTopo --figures comparison,temporal --pdf

    # === Presentation slides (1920x1080 dark dashboard) ===
    python scripts/render_phase5_gifs.py \\
        --ckpt-dir wildfire_phase3_multiseed --out figures/phase5_pres \\
        --seed 42 --regions saudi,california --max-steps 150 \\
        --theme presentation --mp4 --duration 150

    # === Quick smoke test ===
    python scripts/render_phase5_gifs.py \\
        --ckpt-dir wildfire_phase3_multiseed --out figures/phase5 \\
        --seed 42 --regions saudi --max-steps 5 --policies hiercomm_heur,noop

    # === Legacy plain renderer (no basemap) ===
    python scripts/render_phase5_gifs.py \\
        --ckpt-dir wildfire_phase3_multiseed --out figures/phase5 \\
        --seed 42 --no-geo
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from wildfire_marl.env.regimes import make_marl_env
from wildfire_marl.eval.metrics import (
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.eval.phase3_eval import load_nets, select_actions
from wildfire_marl.viz.rollout import compile_rollout_gif, render_rollout_frame, tile_frames_grid

SHOWCASE_POLICIES = ["hiercomm_heur", "local_reactive", "mappo", "noop"]
SHOWCASE_LABELS = {
    "hiercomm_heur": "HierComm (ours)",
    "local_reactive": "Local Reactive",
    "mappo": "MAPPO",
    "noop": "No-Op",
}
EVAL_SEED_OFFSET = 100_000
_COMM_POLICIES = {"hiercomm_heur", "hiercomm", "commnet"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rollout_frames(
    env,
    kind: str,
    nets: dict,
    device: torch.device,
    seed: int,
    region: str,
    max_steps: int,
    renderer=None,
) -> tuple[list[Image.Image], list[float]]:
    """Run one episode, capturing a rendered frame per step.

    Returns (frames, interest_scores) where interest_scores can be used to
    auto-select the most informative snapshot.
    """
    obs, info = env.reset(seed=seed + EVAL_SEED_OFFSET)
    env.apply_target_compliance = False
    done, sc = False, 0
    ce = 1.0
    frames: list[Image.Image] = []
    scores: list[float] = []
    position_history: list[dict[str, tuple[int, int]]] = []
    comm_active = kind in _COMM_POLICIES

    from wildfire_marl.viz.geo_renderer import frame_interest_score

    while not done and sc < max_steps:
        acts = select_actions(env, kind, nets, obs, info, device, sc)
        obs, rew, term, trunc, info = env.step(acts)
        done = term["agent_0"] or trunc["agent_0"]
        ce = info[env.agents[0]].get("coordination_efficiency", ce)

        fs = env.env.fire_state
        wel = weighted_economic_loss(env.env.asset_type, fs, env.env.asset_values)
        isr = infrastructure_survival_rate(env.env.asset_type, fs)

        position_history.append(dict(env.agent_positions))

        if renderer is not None:
            frame = renderer.render_frame(
                fire_state=fs.copy(),
                asset_type=env.env.asset_type,
                agent_positions=env.agent_positions.copy(),
                strategic_targets=env.strategic_targets.copy(),
                step_idx=sc,
                wel=float(wel),
                isr=float(isr),
                ce=float(ce),
                region=region,
                policy_name=SHOWCASE_LABELS.get(kind, kind),
                prev_positions=position_history,
                comm_active=comm_active,
            )
        else:
            frame = render_rollout_frame(
                fire_state=fs.copy(),
                asset_type=env.env.asset_type,
                agent_positions=env.agent_positions.copy(),
                strategic_targets=env.strategic_targets.copy(),
                step_idx=sc,
                wel=float(wel),
                isr=float(isr),
                ce=float(ce),
                region=region,
                policy_name=SHOWCASE_LABELS.get(kind, kind),
            )

        frames.append(frame)
        scores.append(frame_interest_score(fs, env.env.asset_type))
        sc += 1

    return frames, scores


def _best_frame_idx(scores: list[float]) -> int:
    """Select the most visually informative frame index."""
    if not scores:
        return 0
    return int(np.argmax(scores))


def _save_snapshot_png(frames: list[Image.Image], path: Path, step: int = -1,
                       dpi: int = 300) -> None:
    idx = step if step >= 0 else max(0, len(frames) + step)
    idx = min(idx, len(frames) - 1)
    img = frames[idx]
    w, h = img.size
    scale = dpi / 150.0
    new_size = (int(w * scale), int(h * scale))
    img_hr = img.resize(new_size, Image.LANCZOS)
    img_hr.save(path, dpi=(dpi, dpi))


def _save_mp4(frames: list[Image.Image], path: Path, fps: int = 5) -> bool:
    try:
        import imageio.v3 as iio
    except ImportError:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = [np.array(f.convert("RGB")) for f in frames]
    iio.imwrite(str(path), rgb_frames, fps=fps, codec="libx264",
                plugin="pyav", pixelformat="yuv420p")
    return True


def _save_pdf_snapshot(
    renderer, env, kind, nets, device, seed, region, max_steps, step_idx, out_path,
):
    """Re-render a single frame and save as PDF vector."""
    set_seed(seed)
    obs, info = env.reset(seed=seed + EVAL_SEED_OFFSET)
    env.apply_target_compliance = False
    done, sc = False, 0
    ce = 1.0
    position_history: list[dict] = []
    comm_active = kind in _COMM_POLICIES

    while not done and sc <= step_idx:
        acts = select_actions(env, kind, nets, obs, info, device, sc)
        obs, rew, term, trunc, info = env.step(acts)
        done = term["agent_0"] or trunc["agent_0"]
        ce = info[env.agents[0]].get("coordination_efficiency", ce)
        position_history.append(dict(env.agent_positions))

        if sc == step_idx:
            fs = env.env.fire_state
            wel = weighted_economic_loss(env.env.asset_type, fs, env.env.asset_values)
            isr = infrastructure_survival_rate(env.env.asset_type, fs)
            renderer._fire_history.clear()
            renderer._step = -1
            for s in range(sc + 1):
                renderer._update_fire_history(fs, s)
            renderer.render_frame(
                fire_state=fs.copy(),
                asset_type=env.env.asset_type,
                agent_positions=env.agent_positions.copy(),
                strategic_targets=env.strategic_targets.copy(),
                step_idx=sc, wel=float(wel), isr=float(isr), ce=float(ce),
                region=region,
                policy_name=SHOWCASE_LABELS.get(kind, kind),
                prev_positions=position_history,
                comm_active=comm_active,
                save_pdf=str(out_path),
            )
            return
        sc += 1


def _make_temporal_strip(
    all_frames: dict[str, list[Image.Image]],
    steps: list[int],
    out_path: Path,
    dpi: int = 300,
) -> None:
    policy = "hiercomm_heur"
    frames = all_frames.get(policy)
    if not frames:
        return
    ep_len = len(frames)
    valid_steps = [s for s in steps if s < ep_len]
    if len(valid_steps) < len(steps):
        valid_steps.append(ep_len - 1)
    seen = set()
    unique_steps = []
    for s in valid_steps:
        if s not in seen:
            seen.add(s)
            unique_steps.append(s)
    valid_steps = unique_steps
    selected = [frames[s] for s in valid_steps]
    if not selected:
        return
    steps = valid_steps

    n = len(selected)
    w, h = selected[0].size
    pad = 6
    strip_w = n * w + (n - 1) * pad
    canvas = Image.new("RGBA", (strip_w, h), (26, 26, 46, 255))
    for i, img in enumerate(selected):
        canvas.paste(img, (i * (w + pad), 0))

    scale = dpi / 150.0
    canvas = canvas.resize((int(strip_w * scale), int(h * scale)), Image.LANCZOS)
    canvas.save(out_path, dpi=(dpi, dpi))
    print(f"  Temporal strip -> {out_path} ({n} panels at steps {steps})")


def _make_comparison_snapshot(
    all_frames: dict[str, list[Image.Image]],
    step: int,
    out_path: Path,
    dpi: int = 300,
) -> None:
    ordered = [k for k in SHOWCASE_POLICIES if k in all_frames]
    if len(ordered) < 2:
        return
    imgs = [all_frames[k][min(step, len(all_frames[k]) - 1)] for k in ordered]

    cols = 2
    rows = -(-len(imgs) // cols)
    w, h = imgs[0].size
    pad = 6
    grid_w = cols * w + (cols - 1) * pad
    grid_h = rows * h + (rows - 1) * pad
    canvas = Image.new("RGBA", (grid_w, grid_h), (26, 26, 46, 255))
    for j, img in enumerate(imgs):
        r, c = divmod(j, cols)
        canvas.paste(img, (c * (w + pad), r * (h + pad)))

    scale = dpi / 150.0
    canvas = canvas.resize((int(grid_w * scale), int(grid_h * scale)), Image.LANCZOS)
    canvas.save(out_path, dpi=(dpi, dpi))
    print(f"  Comparison snapshot -> {out_path} ({len(imgs)} panels at step {step})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-5 rollout GIF renderer (GIS).")
    ap.add_argument("--ckpt-dir", default="wildfire_phase3_multiseed")
    ap.add_argument("--out", default="figures/phase5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regions", default="saudi,california")
    ap.add_argument("--policies", default=",".join(SHOWCASE_POLICIES))
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--duration", type=int, default=200, help="GIF frame duration (ms)")
    ap.add_argument("--regime", default="default")
    ap.add_argument("--style", default="EsriWorldTopo")
    ap.add_argument("--theme", default="paper", choices=["paper", "presentation"])
    ap.add_argument("--no-geo", action="store_true", help="Use legacy plain renderer")
    ap.add_argument("--snapshot-step", type=int, default=-2,
                    help="Step for PNG snapshot (-1=last, -2=auto-select best)")
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--pdf", action="store_true", help="Also generate PDF vector snapshots")
    ap.add_argument("--data-dir", default="data/cell2fire")
    ap.add_argument("--figures", default="",
                    help="Comma-separated: comparison,temporal")
    ap.add_argument("--temporal-steps", default="20,60,100,140")
    ap.add_argument("--comparison-step", type=int, default=-2,
                    help="Step for comparison figure (-2=auto)")
    args = ap.parse_args()

    device = torch.device("cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = [r.strip() for r in args.regions.split(",")]
    policies = [p.strip() for p in args.policies.split(",")]
    figures = [f.strip() for f in args.figures.split(",") if f.strip()]
    temporal_steps = [int(s) for s in args.temporal_steps.split(",")]

    GeoRenderer = None
    if not args.no_geo:
        try:
            from wildfire_marl.viz.geo_renderer import GeoRenderer as _GR
            GeoRenderer = _GR
            print(f"GIS renderer: {args.style} / theme={args.theme}")
        except ImportError:
            print("WARNING: contextily not installed, falling back to plain renderer")

    for region in regions:
        print(f"\n{'='*60}")
        print(f"  Region: {region.upper()}")
        print(f"{'='*60}")

        renderer = None
        if GeoRenderer is not None:
            renderer = GeoRenderer(region, style=args.style, data_dir=args.data_dir,
                                   theme=args.theme)
            print(f"  Basemap: {renderer.tile_info['name']}")

        all_frame_lists: list[list[Image.Image]] = []
        all_frames_by_policy: dict[str, list[Image.Image]] = {}
        all_scores_by_policy: dict[str, list[float]] = {}

        for kind in policies:
            set_seed(args.seed)
            env = make_marl_env(region, regime=args.regime)

            train_seed = args.seed if kind in {"mappo", "commnet", "hiercomm_heur",
                                                "hiercomm", "qmix"} else None
            try:
                nets = load_nets(kind, region, args.ckpt_dir, env.num_agents, device,
                                 seed=train_seed)
            except FileNotFoundError as e:
                print(f"  SKIP {kind}: {e}")
                env.close()
                continue

            label = SHOWCASE_LABELS.get(kind, kind)
            print(f"  Rendering {label} ({args.max_steps} steps)...", end=" ", flush=True)

            if renderer is not None:
                renderer._fire_history.clear()
                renderer._step = -1

            frames, scores = rollout_frames(
                env, kind, nets, device, args.seed, region, args.max_steps,
                renderer=renderer,
            )
            env.close()

            gif_path = out_dir / f"rollout_{kind}_{region}.gif"
            compile_rollout_gif(frames, gif_path, duration=args.duration)
            print(f"GIF({len(frames)}f)", end="", flush=True)

            if args.snapshot_step == -2:
                snap_idx = _best_frame_idx(scores)
            elif args.snapshot_step == -1:
                snap_idx = len(frames) - 1
            else:
                snap_idx = min(args.snapshot_step, len(frames) - 1)

            png_path = out_dir / f"snapshot_{kind}_{region}.png"
            _save_snapshot_png(frames, png_path, step=snap_idx)
            print(f" PNG(step={snap_idx})", end="", flush=True)

            if args.mp4:
                mp4_path = out_dir / f"rollout_{kind}_{region}.mp4"
                if _save_mp4(frames, mp4_path):
                    print(f" MP4", end="", flush=True)

            print(f" -> {gif_path}")
            all_frame_lists.append(frames)
            all_frames_by_policy[kind] = frames
            all_scores_by_policy[kind] = scores

        comp_step = args.comparison_step
        if comp_step == -2 and "hiercomm_heur" in all_scores_by_policy:
            comp_step = _best_frame_idx(all_scores_by_policy["hiercomm_heur"])
        elif comp_step < 0:
            comp_step = 80
        min_ep_len = min(
            (len(fl) for fl in all_frames_by_policy.values()), default=comp_step + 1
        )
        comp_step = min(comp_step, min_ep_len - 1)

        if "comparison" in figures and all_frames_by_policy:
            _make_comparison_snapshot(
                all_frames_by_policy, comp_step,
                out_dir / f"fig_comparison_{region}.png",
            )

        if "temporal" in figures and all_frames_by_policy:
            _make_temporal_strip(
                all_frames_by_policy, temporal_steps,
                out_dir / f"fig_temporal_{region}.png",
            )

        if len(all_frame_lists) >= 2:
            max_len = max(len(fl) for fl in all_frame_lists)
            for fl in all_frame_lists:
                while len(fl) < max_len:
                    fl.append(fl[-1])
            max_gif_frames = 80
            if max_len > max_gif_frames:
                indices = np.linspace(0, max_len - 1, max_gif_frames, dtype=int).tolist()
                gif_duration = int(args.duration * max_len / max_gif_frames)
            else:
                indices = list(range(max_len))
                gif_duration = args.duration

            print(f"  Composing {len(all_frame_lists)}-panel comparison grid ({len(indices)} frames)...", end=" ", flush=True)
            cols = 2
            rows = -(-len(all_frame_lists) // cols)
            w, h = all_frame_lists[0][0].size
            pad = 4
            grid_w = cols * w + (cols - 1) * pad
            grid_h = rows * h + (rows - 1) * pad

            first_frame = None
            append_frames = []
            for i in indices:
                canvas = Image.new("RGBA", (grid_w, grid_h), (30, 30, 30, 255))
                for j, fl in enumerate(all_frame_lists):
                    r, c = divmod(j, cols)
                    canvas.paste(fl[i], (c * (w + pad), r * (h + pad)))
                if first_frame is None:
                    first_frame = canvas
                else:
                    append_frames.append(canvas)
            grid_path = out_dir / f"comparison_grid_{region}.gif"
            first_frame.save(
                grid_path, save_all=True, append_images=append_frames,
                duration=gif_duration, loop=0,
            )
            del append_frames, first_frame
            print(f"-> {grid_path}")

            snap_step = min(comp_step, max_len - 1)
            snap_canvas = Image.new("RGBA", (grid_w, grid_h), (30, 30, 30, 255))
            for j, fl in enumerate(all_frame_lists):
                r, c = divmod(j, cols)
                snap_canvas.paste(fl[snap_step], (c * (w + pad), r * (h + pad)))
            snap_path = out_dir / f"comparison_snapshot_{region}.png"
            snap_canvas.save(snap_path, dpi=(300, 300))
            print(f"  Comparison snapshot -> {snap_path} (step={snap_step})")
            del snap_canvas

    print("\nPhase 5 GIF rendering complete.")
    print("\n--- Generation commands ---")
    print("# AAAI paper:")
    print(f"python scripts/render_phase5_gifs.py --ckpt-dir {args.ckpt_dir} "
          f"--out figures/phase5 --seed 42 --style EsriWorldTopo "
          f"--figures comparison,temporal --pdf")
    print("# Presentation:")
    print(f"python scripts/render_phase5_gifs.py --ckpt-dir {args.ckpt_dir} "
          f"--out figures/phase5_pres --seed 42 --theme presentation --mp4")


if __name__ == "__main__":
    main()
