"""Deterministic, reusable figure generation.

Replaces ad-hoc plotting cells scattered across notebooks. Each function takes data +
an output path and writes a 300-DPI figure, so ``make figures`` regenerates every paper
figure reproducibly. matplotlib is imported lazily (keeps import-time deps light).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_rl.data.tensor_stack import CHANNEL_ORDER

_DPI = 300


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    return out_path


def plot_channels(state_tensor: np.ndarray, out_path: str | Path):
    """Plot the 7 channels of a state tensor as a 2x4 grid."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    for i, name in enumerate(CHANNEL_ORDER):
        im = axes[i].imshow(state_tensor[i])
        axes[i].set_title(name)
        plt.colorbar(im, ax=axes[i])
    axes[-1].axis("off")
    fig.tight_layout()
    path = _save(fig, out_path)
    plt.close(fig)
    return path


def plot_seed_bars(df: pd.DataFrame, value_col: str, out_path: str | Path, title: str = ""):
    """Bar chart of a metric across seeds (expects columns 'Seed' and ``value_col``)."""
    import matplotlib.pyplot as plt

    seed_col = "Seed" if "Seed" in df.columns else df.columns[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df[seed_col].astype(str), df[value_col])
    ax.set_xlabel("Seed")
    ax.set_ylabel(value_col)
    ax.set_title(title or value_col)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    path = _save(fig, out_path)
    plt.close(fig)
    return path


def plot_transfer_heatmap(df: pd.DataFrame, value: str, out_path: str | Path, title: str = ""):
    """Heatmap of the transfer matrix (rows=train_region, cols=test_region)."""
    import matplotlib.pyplot as plt

    pivot = df.pivot(index="train_region", columns="test_region", values=value)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Test region")
    ax.set_ylabel("Train region")
    ax.set_title(title or f"Transfer: {value}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax)
    path = _save(fig, out_path)
    plt.close(fig)
    return path
