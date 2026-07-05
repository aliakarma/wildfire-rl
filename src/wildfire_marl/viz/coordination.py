"""Visualization tools for multi-agent spatial coordination and trajectory analysis.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_agent_trajectories(
    height: int,
    width: int,
    agent_paths: dict[str, list[tuple[int, int]]],
    fire_state: np.ndarray,
    save_path: str | Path,
):
    """Plot agent trajectories and fire state as a spatial overview."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot fire as red-orange background
    fire_mask = (fire_state > 0).astype(float)
    ax.imshow(fire_mask, cmap="Reds", alpha=0.3, origin="upper")

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    # Plot each agent path with a distinct color
    colors = ["blue", "green", "purple", "orange", "magenta"]
    for idx, (agent, path) in enumerate(agent_paths.items()):
        if not path:
            continue
        path_arr = np.array(path)
        color = colors[idx % len(colors)]

        # Draw trajectory line
        ax.plot(path_arr[:, 1], path_arr[:, 0], color=color, linewidth=2, label=agent)
        # Draw start cell
        ax.scatter(path_arr[0, 1], path_arr[0, 0], color=color, marker="o", s=80, edgecolors="black", zorder=5)
        # Draw end cell
        ax.scatter(path_arr[-1, 1], path_arr[-1, 0], color=color, marker="X", s=100, edgecolors="black", zorder=5)

    ax.set_title("Multi-Agent Suppression Trajectories")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5) # Match standard image coordinates (0,0 top-left)
    ax.legend(loc="upper right")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
