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


def plot_baseline_comparison(eval_csv: str | Path, out_path: str | Path, title: str = ""):
    """Grouped bar chart: PPO vs Random vs NoOp with error bars and significance stars."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(eval_csv)
    baselines = df[df["policy"].isin(["random", "noop"])]
    ppo_rows = df[df["policy"].str.startswith("ppo")]

    ppo_mean = ppo_rows["reward_mean"].mean()
    ppo_std = ppo_rows["reward_mean"].std()
    ppo_err = ppo_std if not np.isnan(ppo_std) else 0.0

    rand_row = baselines[baselines["policy"] == "random"]
    noop_row = baselines[baselines["policy"] == "noop"]

    rand_mean = rand_row["reward_mean"].values[0] if not rand_row.empty else 0.0
    noop_mean = noop_row["reward_mean"].values[0] if not noop_row.empty else 0.0

    rand_err = (
        (rand_row["reward_ci_hi"].values[0] - rand_row["reward_ci_lo"].values[0]) / 2
        if not rand_row.empty
        else 0.0
    )
    noop_err = (
        (noop_row["reward_ci_hi"].values[0] - noop_row["reward_ci_lo"].values[0]) / 2
        if not noop_row.empty
        else 0.0
    )

    policies = ["PPO (Ours)", "Random", "No-Action"]
    means = [ppo_mean, rand_mean, noop_mean]
    yerrs = [ppo_err, rand_err, noop_err]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e"]
    ax.bar(policies, means, yerr=yerrs, capsize=5, color=colors, edgecolor="black", alpha=0.9)

    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(title or "Policy Performance Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Significance stars
    for idx, policy_name in enumerate(["random", "noop"]):
        row_data = baselines[baselines["policy"] == policy_name]
        if not row_data.empty and "sig_vs_ppo" in row_data.columns:
            sig = row_data["sig_vs_ppo"].values[0]
            if sig and sig != "n.s.":
                val = means[idx + 1]
                offset = abs(val) * 0.05
                y_pos = val + offset if val > 0 else val - offset
                ax.text(
                    idx + 1, y_pos, sig, ha="center", va="bottom", fontsize=12, fontweight="bold"
                )

    path = _save(fig, out_path)
    plt.close(fig)
    return path


def plot_ablation_bars(ablation_csv: str | Path, out_path: str | Path):
    """Bar chart with error bars + significance annotations for ablation."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(ablation_csv)
    df["sort_idx"] = df["experiment"].apply(lambda x: 0 if x == "baseline" else 1)
    df = df.sort_values(by=["sort_idx", "experiment"]).reset_index(drop=True)

    yerr = (df["fire_ci_hi"] - df["fire_ci_lo"]) / 2

    labels_map = {
        "baseline": "Baseline (All)",
        "no_wind": "No Wind",
        "no_terrain": "No Terrain",
        "no_suppression": "No Agent (No-Op)",
        "dense_fuel": "Dense Fuel",
    }
    labels = [labels_map.get(x, x) for x in df["experiment"]]
    colors = ["#2ca02c" if x == "baseline" else "#d62728" for x in df["experiment"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        labels, df["fire_mean"], yerr=yerr, capsize=5, color=colors, edgecolor="black", alpha=0.85
    )

    ax.set_ylabel("Mean Fire Intensity")
    ax.set_title("Ablation Study: Impact of Dynamics Components")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add significance stars
    for i, row in df.iterrows():
        variant = row["experiment"]
        if variant == "baseline":
            continue
        sig = row.get("sig_vs_baseline", "")
        if sig and sig != "n.s.":
            y_pos = row["fire_mean"] + yerr[i] + 0.1
            ax.text(labels[i], y_pos, sig, ha="center", va="bottom", fontsize=12, fontweight="bold")

    path = _save(fig, out_path)
    plt.close(fig)
    return path


def plot_marl_scaling(marl_csv: str | Path, out_path: str | Path):
    """Line plot: fire intensity vs number of agents, per region, with CI bands when present.

    Tolerant of the ``run_marl_evaluation`` schema (``fire_intensity_mean``, one row per
    region × team-size) and the older ``fire_mean``/``fire_ci_*`` schema.
    """
    import matplotlib.pyplot as plt

    df = pd.read_csv(marl_csv)
    fire_col = "fire_mean" if "fire_mean" in df.columns else "fire_intensity_mean"
    has_ci = {"fire_ci_lo", "fire_ci_hi"}.issubset(df.columns)
    fig, ax = plt.subplots(figsize=(7, 5))

    regions = list(df["region"].unique()) if "region" in df.columns else [None]
    for reg in regions:
        sub = (df if reg is None else df[df["region"] == reg]).sort_values("num_agents")
        label = str(reg) if reg is not None else "Mean Fire Intensity"
        ax.plot(sub["num_agents"], sub[fire_col], marker="o", linewidth=2, label=label)
        if has_ci:
            ax.fill_between(sub["num_agents"], sub["fire_ci_lo"], sub["fire_ci_hi"], alpha=0.2)

    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Mean Fire Intensity")
    ax.set_title("MARL Cooperative Scaling")
    ax.set_xticks(sorted(df["num_agents"].unique()))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    path = _save(fig, out_path)
    plt.close(fig)
    return path


def plot_generalization_comparison(
    fixed_csv: str | Path, random_csv: str | Path, out_path: str | Path
):
    """Side-by-side comparison: fixed ignition vs randomized ignition performance."""
    import matplotlib.pyplot as plt

    df_fixed = pd.read_csv(fixed_csv)
    df_rand = pd.read_csv(random_csv)

    ppo_fixed = df_fixed[df_fixed["policy"].str.startswith("ppo")]["reward_mean"].mean()
    ppo_fixed_std = df_fixed[df_fixed["policy"].str.startswith("ppo")]["reward_mean"].std()
    ppo_fixed_err = ppo_fixed_std if not np.isnan(ppo_fixed_std) else 0.0

    ppo_rand = df_rand[df_rand["policy"].str.startswith("ppo")]["reward_mean"].mean()
    ppo_rand_std = df_rand[df_rand["policy"].str.startswith("ppo")]["reward_mean"].std()
    ppo_rand_err = ppo_rand_std if not np.isnan(ppo_rand_std) else 0.0

    categories = ["Fixed Ignition (Train)", "Randomized (Generalization)"]
    means = [ppo_fixed, ppo_rand]
    yerrs = [ppo_fixed_err, ppo_rand_err]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        categories,
        means,
        yerr=yerrs,
        capsize=5,
        color=["#2ca02c", "#9467bd"],
        edgecolor="black",
        alpha=0.85,
        width=0.5,
    )

    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Domain Generalization Analysis")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    path = _save(fig, out_path)
    plt.close(fig)
    return path


def plot_state_tensor_comparison(
    saudi_tensor: np.ndarray, ca_tensor: np.ndarray, out_path: str | Path
):
    """Side-by-side 7-channel comparison of both regions (shows ecological difference)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 2, figsize=(10, 20))
    for i, name in enumerate(CHANNEL_ORDER):
        im1 = axes[i, 0].imshow(saudi_tensor[i], cmap="viridis")
        axes[i, 0].set_title(f"Saudi - {name}")
        plt.colorbar(im1, ax=axes[i, 0])

        im2 = axes[i, 1].imshow(ca_tensor[i], cmap="viridis")
        axes[i, 1].set_title(f"California - {name}")
        plt.colorbar(im2, ax=axes[i, 1])

    fig.tight_layout()
    path = _save(fig, out_path)
    plt.close(fig)
    return path
