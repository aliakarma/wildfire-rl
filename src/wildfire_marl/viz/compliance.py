"""Visualization tools for strategic target compliance.
Provides trajectory overlays, sector occupancy heatmaps, and compliance plots over time.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def compute_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def plot_trajectory_overlay(
    positions_history: dict[str, list[tuple[int, int]]],
    targets_history: dict[str, list[tuple[int, int]]],
    save_path: str | Path,
):
    """Plot trajectories of agents and their targets on a 2D grid."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw a grid background representing the 32x32 area
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(-0.5, 31.5)
    ax.set_xticks(np.arange(0, 32, 8))
    ax.set_yticks(np.arange(0, 32, 8))
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    
    colors = ["red", "blue", "green"]
    for i, (agent, pos) in enumerate(positions_history.items()):
        color = colors[i % len(colors)]
        y_pos = [p[0] for p in pos]
        x_pos = [p[1] for p in pos]
        
        # Plot agent path
        ax.plot(x_pos, y_pos, color=color, label=f"{agent} Path", alpha=0.7, linewidth=2)
        ax.scatter(x_pos[0], y_pos[0], color=color, marker='o', s=80, edgecolors='black', zorder=5) # Start
        ax.scatter(x_pos[-1], y_pos[-1], color=color, marker='x', s=80, zorder=5) # End
        
        # Plot target path if present
        tar = targets_history.get(agent, [])
        if tar:
            t_y = [t[0] for t in tar]
            t_x = [t[1] for t in tar]
            ax.plot(t_x, t_y, color=color, linestyle=':', alpha=0.5, linewidth=1.5)
            # Mark target change points
            change_indices = [0] + [j for j in range(1, len(tar)) if tar[j] != tar[j-1]]
            for idx in change_indices:
                ax.scatter(t_x[idx], t_y[idx], color=color, marker='*', s=120, edgecolors='black', zorder=4)

    ax.set_title("Agent Trajectories vs. Assigned Targets")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend(loc="upper right")
    fig.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_sector_occupancy(
    sector_occupancy: dict[int, float],
    save_path: str | Path,
):
    """Plot a 4x4 heatmap representing sector occupancy proportions."""
    grid = np.zeros((4, 4))
    for k, v in sector_occupancy.items():
        y = k // 4
        x = k % 4
        grid[y, x] = v
        
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(grid, cmap="Oranges", origin="upper", vmin=0.0)
    
    # Annotate with percentages
    for i in range(4):
        for j in range(4):
            ax.text(
                j, i, f"{grid[i, j]*100:.1f}%",
                ha="center", va="center",
                color="black" if grid[i, j] < 0.3 else "white",
                fontweight="bold"
            )
            
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels([f"Col {c}" for c in range(4)])
    ax.set_yticklabels([f"Row {r}" for r in range(4)])
    
    ax.set_title("Sector Occupancy Heatmap (4x4)")
    fig.colorbar(im, ax=ax, label="Occupancy Ratio")
    fig.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_distance_to_target(
    positions_history: dict[str, list[tuple[int, int]]],
    targets_history: dict[str, list[tuple[int, int]]],
    save_path: str | Path,
):
    """Plot distance to target over time for each agent."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    colors = ["red", "blue", "green"]
    for i, (agent, pos) in enumerate(positions_history.items()):
        color = colors[i % len(colors)]
        tar = targets_history.get(agent, [])
        if not tar:
            continue
            
        distances = [compute_distance(pos[t], tar[t]) for t in range(len(pos))]
        ax.plot(distances, color=color, label=agent, alpha=0.8, linewidth=2)
        
    ax.set_title("Distance to Assigned Target Over Time")
    ax.set_xlabel("Environment Step")
    ax.set_ylabel("Euclidean Distance")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
