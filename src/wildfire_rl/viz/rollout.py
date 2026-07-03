"""Canonical animated-rollout renderer (Phase 15B.5 / 15B.8).

Every experiment/ablation rollout is rendered in one fixed format so results are visually comparable
across groups, regions, and the native/transfer axis: a per-timestep grid with a ``Timestep #`` header
and the fixed legend (Grass / Fire / Populated / Evacuating / Path / Finished), palette locked in
``configs/viz.yaml``. Wildfire semantics are mapped onto the canonical class names:

  Grass=unburned · Fire=burning · Populated=intact asset · Evacuating=agent cell ·
  Path=agent route (visited) · Finished=extinguished/protected cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_PALETTE_PATH = Path("configs/viz.yaml")


@dataclass
class Palette:
    names: list[str]
    colors: list[str]
    cell_px: int
    fps: int

    @classmethod
    def load(cls, path: str | Path = _PALETTE_PATH) -> Palette:
        cfg = yaml.safe_load(Path(path).read_text())
        classes = sorted(cfg["classes"], key=lambda c: c["code"])
        return cls(
            names=[c["name"] for c in classes],
            colors=[c["color"] for c in classes],
            cell_px=int(cfg.get("cell_px", 14)),
            fps=int(cfg.get("fps", 4)),
        )


# Canonical class codes (must match configs/viz.yaml order).
GRASS, FINISHED, PATH, POPULATED, EVACUATING, FIRE = range(6)
FIRE_THRESHOLD = 0.1


def class_grid(
    fire: np.ndarray,
    asset_type: np.ndarray | None,
    agent_positions,
    visited: np.ndarray | None,
    ever_burned: np.ndarray | None,
) -> np.ndarray:
    """Map the environment state to canonical class codes (higher code wins)."""
    h, w = fire.shape
    g = np.full((h, w), GRASS, dtype=np.int32)
    if ever_burned is not None:
        g[(ever_burned) & (fire <= FIRE_THRESHOLD)] = FINISHED  # burned then extinguished
    if visited is not None:
        g[visited] = PATH
    if asset_type is not None:
        g[asset_type > 0] = POPULATED
    for x, y in agent_positions:
        g[int(x), int(y)] = EVACUATING
    g[fire > FIRE_THRESHOLD] = FIRE  # fire always shows on top
    return g


def record_rollout(env, policy, seed: int = 0, max_steps: int = 60) -> list[dict]:
    """Run one episode, returning per-step frames (class grid + timestep + fire total)."""
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "env"):
        policy.env = env
    positions = _agent_positions(env)
    visited = np.zeros((env.grid_size, env.grid_size), dtype=bool)
    ever = env.state[0] > FIRE_THRESHOLD
    frames = []
    for t in range(1, max_steps + 1):
        for x, y in positions:
            visited[int(x), int(y)] = True
        frames.append(
            {
                "t": t,
                "grid": class_grid(
                    env.state[0], getattr(env, "infra_asset_type", None), positions, visited, ever
                ),
                "fire_total": float(env.state[0].sum()),
            }
        )
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        positions = _agent_positions(env)
        ever = ever | (env.state[0] > FIRE_THRESHOLD)
        if term or trunc:
            break
    return frames


def _agent_positions(env):
    if hasattr(env, "agent_positions"):
        return [tuple(p) for p in env.agent_positions]
    return [tuple(env.agent_pos)]


def record_episode(env, policy, seed: int = 0, max_steps: int = 60) -> dict:
    """Record one deterministic episode's raw fire fields + agent path (Phase 17).

    Returns ``{frames: [fire2d, ...], agent_paths: [[(x,y), ...], ...], criticality, seed}`` for the
    static multi-panel renderer. ``agent_paths[t]`` is the list of agent cells at step ``t``.
    """
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "env"):
        policy.env = env
    frames, agent_paths = [], []
    for _ in range(max_steps):
        frames.append(env.state[0].copy())
        agent_paths.append(_agent_positions(env))
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            frames.append(env.state[0].copy())
            agent_paths.append(_agent_positions(env))
            break
    return {
        "frames": frames,
        "agent_paths": agent_paths,
        "criticality": getattr(env, "infra_criticality", None),
        "seed": seed,
    }


def render_rollout(record: dict, out_path: str | Path, title: str = "", n_panels: int = 5):
    """Multi-panel PNG: fire snapshots at key timesteps + agent trajectory + criticality overlay.

    Headless (Agg). Snapshots at t = 0 / quarters / end; the agent trajectory is drawn up to each
    panel's timestep; the Saudi criticality map is contoured underneath when present.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = record["frames"]
    paths = record["agent_paths"]
    crit = record.get("criticality")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    idxs = np.linspace(0, len(frames) - 1, min(n_panels, len(frames))).astype(int)
    fig, axes = plt.subplots(1, len(idxs), figsize=(3 * len(idxs), 3.3))
    if len(idxs) == 1:
        axes = [axes]
    for ax, k in zip(axes, idxs, strict=False):
        if crit is not None:
            ax.contour(crit, levels=[0.3, 0.6], colors="#0E2A3B", linewidths=0.8, alpha=0.7)
        ax.imshow(frames[k], cmap="hot", vmin=0, vmax=1)  # fire field
        # agent trajectory up to this timestep
        xs = [p[0] for step in paths[: k + 1] for p in [step[0]]]
        ys = [p[1] for step in paths[: k + 1] for p in [step[0]]]
        if len(xs) > 1:
            ax.plot(ys, xs, "-", color="#1F7A9E", linewidth=1.2, alpha=0.8)
        for x, y in paths[k]:
            ax.scatter([y], [x], s=25, c="#1F7A9E", marker="o", edgecolors="w", linewidths=0.5)
        ax.set_title(f"t={k}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_gif(
    frames: list[dict], out_path: str | Path, title: str = "", palette: Palette | None = None
):
    """Render frames to an animated GIF in the canonical format (Timestep header + fixed legend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import ListedColormap

    palette = palette or Palette.load()
    cmap = ListedColormap(palette.colors)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frames[0]["grid"], cmap=cmap, vmin=0, vmax=len(palette.colors) - 1)
    ax.set_xticks([])
    ax.set_yticks([])
    header = ax.set_title(f"{title}   Timestep #: {frames[0]['t']}", fontsize=10, loc="left")
    handles = [
        mpatches.Patch(color=c, label=n)
        for n, c in zip(palette.names, palette.colors, strict=False)
    ]
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False
    )

    def update(i):
        im.set_data(frames[i]["grid"])
        header.set_text(f"{title}   Timestep #: {frames[i]['t']}")
        return [im, header]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=palette.fps))
    plt.close(fig)
    return out_path
