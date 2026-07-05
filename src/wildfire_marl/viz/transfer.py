"""Visualization tools for cross-domain transfer heatmaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_transfer_heatmap(
    matrix_2x2: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    save_path: str | Path,
):
    """Plot a 2x2 Transfer Robustness Score (TRS) heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix_2x2, cmap="YlGn", vmin=0.0, vmax=1.0)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(
                j,
                i,
                f"{matrix_2x2[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if matrix_2x2[i, j] < 0.8 else "white",
                fontweight="bold",
            )

    ax.set_title(title)
    ax.set_xlabel("Target Evaluation Region")
    ax.set_ylabel("Source Training Region")
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
