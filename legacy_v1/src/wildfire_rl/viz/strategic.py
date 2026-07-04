"""Strategic rollout overlays (Phase 15B.5): make what the system defends legible.

Produces a filmstrip PNG (key timesteps side-by-side) with the petroleum-criticality map underlaid,
asset markers, the agent dispatch trajectory, and the fire — so infrastructure-defense behavior and
heuristic-contains-vs-collapse are visually unambiguous. Complements the canonical GIF in
``viz/rollout.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wildfire_rl.viz.rollout import FIRE_THRESHOLD, _agent_positions


def _save(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def plot_strategic_filmstrip(
    env, policy, out_path, seed: int = 0, n_panels: int = 5, max_steps: int = 60
):
    """Render a strategic-rollout filmstrip: criticality underlay + assets + agent trail + fire."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "env"):
        policy.env = env
    crit = getattr(env, "infra_criticality", None)
    assets = getattr(env, "infra_asset_type", None)
    trail = np.zeros((env.grid_size, env.grid_size), dtype=bool)

    snapshots, positions = [], _agent_positions(env)
    for t in range(1, max_steps + 1):
        for x, y in positions:
            trail[int(x), int(y)] = True
        snapshots.append((t, env.state[0].copy(), trail.copy(), list(positions)))
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        positions = _agent_positions(env)
        if term or trunc:
            break

    idxs = np.linspace(0, len(snapshots) - 1, min(n_panels, len(snapshots))).astype(int)
    fig, axes = plt.subplots(1, len(idxs), figsize=(3 * len(idxs), 3.2))
    if len(idxs) == 1:
        axes = [axes]
    for ax, k in zip(axes, idxs, strict=False):
        t, fire, tr, pos = snapshots[k]
        if crit is not None:
            ax.imshow(crit, cmap="Greys", vmin=0, vmax=1, alpha=0.6)  # criticality underlay
        fire_rgba = np.zeros((*fire.shape, 4))
        fire_rgba[fire > FIRE_THRESHOLD] = [0.94, 0.29, 0.43, 0.9]  # fire (pink)
        ax.imshow(fire_rgba)
        tr_rgba = np.zeros((*tr.shape, 4))
        tr_rgba[tr] = [0.96, 0.76, 0.29, 0.5]  # dispatch trail (yellow)
        ax.imshow(tr_rgba)
        if assets is not None:
            ay, ax_ = np.where(assets > 0)
            ax.scatter(
                ax_, ay, s=40, marker="s", facecolors="none", edgecolors="#0E2A3B", linewidths=1.5
            )
        for x, y in pos:
            ax.scatter([y], [x], s=30, c="#1F7A9E", marker="o")  # agents
        ax.set_title(f"t={t}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Strategic rollout: criticality (grey) · fire (pink) · dispatch (yellow) · assets (□) · agents (●)",
        fontsize=9,
    )
    path = _save(fig, out_path)
    plt.close(fig)
    return path
