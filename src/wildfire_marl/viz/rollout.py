"""Visualization tools for canonical rollout rendering and GIF compilation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from wildfire_marl.viz.basemap import fetch_region_basemap


def draw_grid_cell(ax, y, x, color, alpha=1.0, marker=None, size=80, label=None):
    """Helper to draw a specific grid cell on the plot."""
    if marker:
        ax.scatter(
            x, y, color=color, marker=marker, s=size, edgecolors="black", zorder=5, label=label
        )
    else:
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1.0, 1.0, color=color, alpha=alpha, zorder=3)
        ax.add_patch(rect)


def render_rollout_frame(
    fire_state: np.ndarray,
    asset_type: np.ndarray | None,
    agent_positions: dict[str, tuple[int, int]],
    strategic_targets: dict[str, tuple[int, int]],
    step_idx: int,
    wel: float,
    isr: float,
    ce: float,
    region: str,
    policy_name: str,
    use_basemap: bool = True,
) -> Image.Image:
    """Renders a single frame of the environment rollout as a PIL Image."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=100)

    height, width = fire_state.shape
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)  # Top-left is (0,0)

    # 1. Base landscape: OSM basemap for the region ROI when available, else flat grey
    basemap = fetch_region_basemap(region) if use_basemap else None
    if basemap is not None:
        ax.imshow(
            basemap,
            origin="upper",
            extent=(-0.5, width - 0.5, height - 0.5, -0.5),
            alpha=0.75,
            zorder=1,
        )
    else:
        ax.imshow(np.ones((height, width, 3)) * 0.9, origin="upper")

    # Draw 4x4 sector boundaries (16 sectors)
    for i in range(1, 4):
        ax.axhline(i * 8 - 0.5, color="blue", linestyle="--", alpha=0.3, linewidth=1.5)
        ax.axvline(i * 8 - 0.5, color="blue", linestyle="--", alpha=0.3, linewidth=1.5)

    # 2. Draw Fire (black/dark red) and Treated Cells (cyan)
    fire_y, fire_x = np.where(fire_state > 0)
    for y, x in zip(fire_y, fire_x, strict=False):
        draw_grid_cell(ax, y, x, color="#3b0606", alpha=0.85)  # Dark charcoal red

    treated_y, treated_x = np.where(fire_state < 0)
    for y, x in zip(treated_y, treated_x, strict=False):
        draw_grid_cell(ax, y, x, color="#00F0FF", alpha=0.9)  # Neon Cyan

    # 3. Draw Assets
    if asset_type is not None:
        asset_y, asset_x = np.where(asset_type > 0)
        for y, x in zip(asset_y, asset_x, strict=False):
            # Red for refineries (Saudi), Orange for WUI (California)
            color = "#FF1744" if region.lower() == "saudi" else "#FF9100"
            draw_grid_cell(ax, y, x, color=color, marker="*", size=130)

    # 4. Draw Agents and their strategic targets
    colors = ["#1E88E5", "#43A047", "#8E24AA"]  # Blue, Green, Purple
    for i, (agent, pos) in enumerate(agent_positions.items()):
        color = colors[i % len(colors)]
        # Draw Agent
        draw_grid_cell(ax, pos[0], pos[1], color=color, marker="o", size=110)

        # Draw target line and target center
        tar = strategic_targets.get(agent)
        if tar:
            ax.plot(
                [pos[1], tar[1]],
                [pos[0], tar[0]],
                color=color,
                linestyle=":",
                alpha=0.6,
                linewidth=1.5,
            )
            ax.scatter(tar[1], tar[0], color=color, marker="x", s=90, linewidth=2, zorder=4)

    # Title & Text Overlay (AAAI Style)
    title = f"{policy_name} Rollout ({region.capitalize()})"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    info_text = f"Step: {step_idx:02d} | WEL: {wel:.1f} | ISR: {isr:.2f} | CE: {ce:.2f}"
    ax.text(
        0.5,
        -0.07,
        info_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#F5F5F5",
            "edgecolor": "#E0E0E0",
            "alpha": 0.9,
        },
        fontsize=10,
        fontweight="medium",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    # Convert matplotlib figure to PIL Image
    fig.canvas.draw()
    rgba = fig.canvas.buffer_rgba()
    img = Image.frombytes("RGBA", fig.canvas.get_width_height(), rgba)
    plt.close(fig)
    return img


def compile_rollout_gif(
    frames: list[Image.Image],
    save_path: str | Path,
    duration: int = 200,
):
    """Compiles list of PIL images into an animated GIF."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    # Quantize each RGBA frame to a full-frame palette image; without this PIL
    # writes lossy delta frames that ghost/tear on detailed (basemap) backgrounds.
    palettized = [f.convert("RGB").quantize(colors=256, dither=Image.FLOYDSTEINBERG) for f in frames]
    palettized[0].save(
        save_path,
        save_all=True,
        append_images=palettized[1:],
        duration=duration,
        loop=0,
        disposal=2,
        optimize=False,
    )
