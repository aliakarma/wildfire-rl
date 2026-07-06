"""Visualization tools for strategic commander interpretability, including decision arrows, threat heatmaps, and priority grids."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_commander_decisions(
    fire_state: np.ndarray,
    criticality: np.ndarray | None,
    agent_positions: dict[str, tuple[int, int]],
    strategic_targets: dict[str, tuple[int, int]],
    save_path: str | Path,
):
    """Plot commander dispatch decisions as arrows overlaying fire and assets grids."""
    fig, ax = plt.subplots(figsize=(6, 6))

    height, width = fire_state.shape
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

    # Background: fuel / asset criticality
    if criticality is not None:
        ax.imshow(criticality, cmap="YlOrRd", alpha=0.3)
    else:
        ax.imshow(np.zeros((height, width)), cmap="Greens", alpha=0.1)

    # Draw fire cells
    fire_y, fire_x = np.where(fire_state > 0)
    ax.scatter(fire_x, fire_y, color="#3b0606", marker="s", s=10, alpha=0.5, label="Active Fire")

    # Draw 4x4 sector boundaries
    for i in range(1, 4):
        ax.axhline(i * 8 - 0.5, color="black", linestyle=":", alpha=0.3)
        ax.axvline(i * 8 - 0.5, color="black", linestyle=":", alpha=0.3)

    colors = ["#1E88E5", "#43A047", "#8E24AA"]
    for i, (agent, pos) in enumerate(agent_positions.items()):
        color = colors[i % len(colors)]
        target = strategic_targets.get(agent)

        # Agent position marker
        ax.scatter(
            pos[1],
            pos[0],
            color=color,
            marker="o",
            s=100,
            edgecolors="black",
            zorder=5,
            label=f"{agent} Start",
        )

        if target:
            # Target center marker
            ax.scatter(
                target[1], target[0], color=color, marker="*", s=120, edgecolors="black", zorder=5
            )
            # Arrow from agent to target
            dy = target[0] - pos[0]
            dx = target[1] - pos[1]
            ax.arrow(
                pos[1],
                pos[0],
                dx * 0.8,
                dy * 0.8,
                head_width=1.0,
                head_length=1.2,
                fc=color,
                ec=color,
                alpha=0.8,
                length_includes_head=True,
                zorder=4,
            )

    ax.set_title("Strategic Commander Dispatch Decisions")
    ax.set_xlabel("X Grid")
    ax.set_ylabel("Y Grid")
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_priority_heatmaps(
    sector_fire: np.ndarray,
    sector_assets: np.ndarray,
    save_path: str | Path,
):
    """Plot side-by-side heatmaps of commander threat (fire) and value (assets) inputs."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # 1. Sector Fire Heatmap
    im1 = axes[0].imshow(sector_fire.reshape(4, 4), cmap="Reds", origin="upper")
    axes[0].set_title("Sector Threat Grid (Active Fire)")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Sector Assets Heatmap
    im2 = axes[1].imshow(sector_assets.reshape(4, 4), cmap="Oranges", origin="upper")
    axes[1].set_title("Sector Criticality Grid (Asset Priority)")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
