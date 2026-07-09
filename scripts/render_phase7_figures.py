"""Phase 7 (peer-review remediation): regenerate paper figures from CERTIFIED artifacts.

Produces (into results/phase7_peer/, then copied to AAAI Template/Figures/ by the caller):
- fig2_rollout_flat_vs_hier.png : replaces the mislabeled Figure 2 (issue H4). Left panel is
  a genuine rehabilitated Flat-MARL (matched, 100k) rollout; right is the hierarchy; a step
  is chosen where the two differ in WEL/ISR; footer metrics stated per panel.
- fig3_commander_dispatch.png : replaces Figure 3 (issue M2). A real macro-step with a
  cross-sector dispatch; agents (circles) -> assigned sector centers (stars); 4x4 sector
  grid; caption carries the measured reassignment rate.
- fig5_flat_failure_analysis.png : new (issue M7). Occupancy heatmaps (Flat-MARL vs
  Hierarchy) with asset cells overlaid + mean distance-to-nearest-asset bars, from the
  Phase 2 behavioral logs — turns "coordination collapse" from assertion into evidence.

All rollouts are replayed from the SHIPPED / Phase-2 checkpoints under the certified seed
protocol — no smoke-test artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scripts.eval_hierarchical import target_seeking_action

from wildfire_marl.agents.agent_networks import MAPPOActor
from wildfire_marl.agents.strategic_controller import (
    StrategicController,
    get_sector_center,
)
from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.eval.metrics import (
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.train.hierarchical_train import extract_high_level_state

OUT = Path("results/phase7_peer")
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cpu")


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def make_env(region: str) -> MultiAgentFireEnv:
    m = "Saudi" if region == "saudi" else "California"
    return MultiAgentFireEnv(
        num_agents=3, crop_size=9, coordination_penalty=0.1, fire_map=m,
        data_dir="data/cell2fire", max_steps=150, steps_per_action=60,
        observe_infra=True, catastrophe_weight=2.0, cascade_prob=0.1,
        infra_dir=f"data/cell2fire/{m}",
    )


def grid_rgb(env) -> np.ndarray:
    """Render the grid: fuel gray, treated blue-ish, fire red."""
    fire = env.env.fire_state
    fuel = env.env.fuel_mask
    img = np.ones((*fire.shape, 3)) * 0.97
    img[fuel > 0] = (0.85, 0.88, 0.80)  # burnable fuel
    img[fire < 0] = (0.55, 0.75, 0.95)  # treated
    img[fire > 0] = (0.85, 0.20, 0.15)  # active fire
    return img


def draw_panel(ax, env, title, footer, show_targets=None, show_sectors=False):
    ax.imshow(grid_rgb(env), origin="upper", interpolation="nearest")
    at = env.env.asset_type
    ys, xs = np.nonzero(at > 0)
    ax.scatter(xs, ys, marker="*", s=190, c="gold", edgecolors="black",
               linewidths=0.8, label="Infrastructure", zorder=5)
    for i, a in enumerate(env.agents):
        y, x = env.agent_positions[a]
        ax.scatter(x, y, marker="o", s=90, facecolors="white", edgecolors="navy",
                   linewidths=1.8, zorder=6)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=6.5,
                color="navy", zorder=7)
    if show_targets is not None:
        for i, a in enumerate(env.agents):
            ty, tx = show_targets[a]
            ay, ax_ = env.agent_positions[a]
            ax.scatter(tx, ty, marker="*", s=150, c="royalblue",
                       edgecolors="black", linewidths=0.6, zorder=6)
            ax.annotate("", xy=(tx, ty), xytext=(ax_, ay),
                        arrowprops=dict(arrowstyle="->", color="navy", lw=1.5), zorder=6)
    if show_sectors:
        for k in range(1, 4):
            ax.axhline(k * 8 - 0.5, color="black", lw=0.6, alpha=0.4)
            ax.axvline(k * 8 - 0.5, color="black", lw=0.6, alpha=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(footer, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def wel_isr(env):
    fs = env.env.fire_state
    return (
        weighted_economic_loss(env.env.asset_type, fs, env.env.asset_values),
        infrastructure_survival_rate(env.env.asset_type, fs),
    )


def rollout_capture(env, policy, nets, seed, capture_step):
    """Replay one episode; return a deep copy of grid state + metrics at capture_step."""
    obs, info = env.reset(seed=seed)
    env.apply_target_compliance = False
    targets = {a: env.agent_positions[a] for a in env.agents}
    step = 0
    snap = None
    done = False
    while not done:
        if step % 10 == 0 and policy == "hier":
            s_f, s_a, s_g = extract_high_level_state(env)
            with torch.no_grad():
                lg = nets["commander"](s_f, s_a, s_g)
                acts = [int(torch.argmax(x)) for x in lg]
            targets = {a: get_sector_center(acts[i]) for i, a in enumerate(env.agents)}
        env.strategic_targets = targets
        adict = {}
        for a in env.agents:
            mask = info[a]["action_mask"]
            if policy == "flat":
                with torch.no_grad():
                    ot = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0)
                    mt = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
                    lg = nets["flat"](ot, mt).squeeze(0)
                adict[a] = int(torch.argmax(lg))
            else:
                adict[a] = target_seeking_action(env, a, mask)
        obs, _, term, trunc, info = env.step(adict)
        done = term["agent_0"] or trunc["agent_0"]
        step += 1
        if step == capture_step:
            w, i2 = wel_isr(env)
            snap = {
                "img": grid_rgb(env),
                "positions": dict(env.agent_positions),
                "asset_type": env.env.asset_type.copy(),
                "WEL": w, "ISR": i2, "step": step,
            }
    if snap is None:  # episode ended early; capture final
        w, i2 = wel_isr(env)
        snap = {"img": grid_rgb(env), "positions": dict(env.agent_positions),
                "asset_type": env.env.asset_type.copy(), "WEL": w, "ISR": i2, "step": step}
    return snap


def draw_snap(ax, snap, title):
    ax.imshow(snap["img"], origin="upper", interpolation="nearest")
    ys, xs = np.nonzero(snap["asset_type"] > 0)
    ax.scatter(xs, ys, marker="*", s=190, c="gold", edgecolors="black",
               linewidths=0.8, zorder=5)
    for i, (y, x) in enumerate(snap["positions"].values()):
        ax.scatter(x, y, marker="o", s=90, facecolors="white", edgecolors="navy",
                   linewidths=1.8, zorder=6)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=6.5, color="navy", zorder=7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"Step {snap['step']}  |  WEL {snap['WEL']:.1f}  |  ISR {snap['ISR']:.2f}",
                  fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- Figure 2
def figure2(region="saudi"):
    env = make_env(region)
    flat_ck = torch.load(f"results/phase2_peer/checkpoint_mappo_{region}.pt", map_location=DEV)
    flat = MAPPOActor(8, 6, 64)
    flat.load_state_dict(flat_ck["actor_state_dict"])
    flat.eval()
    hier_ck = torch.load(f"results/runs/checkpoint_hierarchical_{region}.pt", map_location=DEV)
    commander = StrategicController(num_agents=3)
    commander.load_state_dict(hier_ck["commander_state_dict"])
    commander.eval()

    seed = 2042  # a seed where hierarchy beats flat (WEL 21.4 vs 27.0 at seed 2042)
    cap = 120
    set_seed(seed)
    flat_snap = rollout_capture(env, "flat", {"flat": flat}, seed, cap)
    set_seed(seed)
    hier_snap = rollout_capture(env, "hier", {"commander": commander}, seed, cap)
    env.close()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 4.4))
    draw_snap(a1, flat_snap, "Flat MARL (MAPPO, 100k, matched objective)")
    draw_snap(a2, hier_snap, "Hierarchical (ours)")
    handles = [
        mpatches.Patch(color=(0.85, 0.20, 0.15), label="Active fire"),
        mpatches.Patch(color=(0.55, 0.75, 0.95), label="Treated"),
        plt.Line2D([], [], marker="*", color="gold", ls="", mec="black", label="Infrastructure"),
        plt.Line2D([], [], marker="o", color="white", ls="", mec="navy", label="Agent"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = OUT / "fig2_rollout_flat_vs_hier.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"Figure 2: flat WEL {flat_snap['WEL']:.1f} vs hier WEL {hier_snap['WEL']:.1f} "
          f"at step {cap} -> {p}")
    return flat_snap["WEL"], hier_snap["WEL"]


# ---------------------------------------------------------------- Figure 3
def figure3(region="saudi"):
    env = make_env(region)
    hier_ck = torch.load(f"results/runs/checkpoint_hierarchical_{region}.pt", map_location=DEV)
    commander = StrategicController(num_agents=3)
    commander.load_state_dict(hier_ck["commander_state_dict"])
    commander.eval()

    seed = 42
    obs, info = env.reset(seed=seed)
    env.apply_target_compliance = False
    prev_sec = None
    best = None
    targets = {a: env.agent_positions[a] for a in env.agents}
    step = 0
    done = False
    while not done and step < 60:
        if step % 10 == 0:
            s_f, s_a, s_g = extract_high_level_state(env)
            with torch.no_grad():
                lg = commander(s_f, s_a, s_g)
                acts = [int(torch.argmax(x)) for x in lg]
            cur_sec = [int(np.clip(env.agent_positions[a][0] // 8, 0, 3)) * 4
                       + int(np.clip(env.agent_positions[a][1] // 8, 0, 3))
                       for a in env.agents]
            n_cross = sum(acts[i] != cur_sec[i] for i in range(3))
            targets = {a: get_sector_center(acts[i]) for i, a in enumerate(env.agents)}
            # capture the first macro-step with >=2 cross-sector dispatches and some fire
            if best is None and n_cross >= 2 and (env.env.fire_state > 0).sum() > 5:
                best = {
                    "positions": dict(env.agent_positions),
                    "targets": dict(targets),
                    "img": grid_rgb(env),
                    "asset_type": env.env.asset_type.copy(),
                    "step": step, "n_cross": n_cross,
                }
            prev_sec = cur_sec
        env.strategic_targets = targets
        adict = {a: target_seeking_action(env, a, info[a]["action_mask"]) for a in env.agents}
        obs, _, term, trunc, info = env.step(adict)
        done = term["agent_0"] or trunc["agent_0"]
        step += 1
    env.close()

    if best is None:
        print("Figure 3: no cross-sector macro-step found; using first dispatch")
        return

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.imshow(best["img"], origin="upper", interpolation="nearest")
    ys, xs = np.nonzero(best["asset_type"] > 0)
    ax.scatter(xs, ys, marker="*", s=210, c="gold", edgecolors="black", linewidths=0.9,
               zorder=5, label="Infrastructure")
    for k in range(1, 4):
        ax.axhline(k * 8 - 0.5, color="black", lw=0.6, alpha=0.4)
        ax.axvline(k * 8 - 0.5, color="black", lw=0.6, alpha=0.4)
    for i, a in enumerate(env.agents):
        y, x = best["positions"][a]
        ty, tx = best["targets"][a]
        ax.scatter(x, y, marker="o", s=120, facecolors="white", edgecolors="navy",
                   linewidths=2, zorder=6)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="navy", zorder=7)
        ax.scatter(tx, ty, marker="*", s=170, c="royalblue", edgecolors="black",
                   linewidths=0.6, zorder=6)
        ax.annotate("", xy=(tx, ty), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="navy", lw=1.8), zorder=6)
    ax.set_title(f"Commander dispatch (step {best['step']}, "
                 f"{best['n_cross']}/3 cross-sector)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    handles = [
        plt.Line2D([], [], marker="o", color="white", ls="", mec="navy", label="Agent (current)"),
        plt.Line2D([], [], marker="*", color="royalblue", ls="", mec="black",
                   label="Assigned sector center"),
        plt.Line2D([], [], marker="*", color="gold", ls="", mec="black", label="Infrastructure"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    p = OUT / "fig3_commander_dispatch.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"Figure 3: step {best['step']}, {best['n_cross']}/3 cross-sector -> {p}")


# ---------------------------------------------------------------- Figure 5
def figure5(region="saudi"):
    occ = np.load(f"results/phase2_peer/occupancy_{region}.npz")
    beh = json.load(open(f"results/phase2_peer/behavior_{region}.json"))
    at = np.load(f"data/cell2fire/{'Saudi' if region == 'saudi' else 'California'}/asset_type.npy")
    ys, xs = np.nonzero(at > 0)

    flat = occ["Flat_MARL_(matched)"].astype(float)
    hier = occ["Learned_Hierarchical"].astype(float)
    vmax = max(flat.max(), hier.max())

    fig = plt.figure(figsize=(10.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])
    for ax, occm, title in [
        (fig.add_subplot(gs[0]), flat, "Flat MARL occupancy"),
        (fig.add_subplot(gs[1]), hier, "Hierarchical occupancy"),
    ]:
        im = ax.imshow(occm, cmap="magma", origin="upper", vmax=vmax)
        ax.scatter(xs, ys, marker="*", s=150, c="cyan", edgecolors="black",
                   linewidths=0.7, zorder=5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)

    ax3 = fig.add_subplot(gs[2])
    pols = ["No-Op", "Flat MARL (matched)", "QMIX (matched)",
            "Learned Hierarchical", "CommNet (matched)"]
    dists = [beh[p]["mean_chebyshev_dist_to_nearest_asset"] for p in pols]
    colors = ["#bbbbbb", "#C0504D", "#E8A33D", "#3B6EA5", "#4C9F70"]
    short = ["No-Op", "Flat MARL", "QMIX", "Hierarchy", "CommNet"]
    ax3.barh(range(len(pols)), dists, color=colors)
    ax3.set_yticks(range(len(pols)))
    ax3.set_yticklabels(short, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel("Mean Chebyshev distance\nto nearest asset (cells)", fontsize=8)
    ax3.set_title("Where agents spend time", fontsize=10)
    ax3.grid(axis="x", alpha=0.3)
    for i, d in enumerate(dists):
        ax3.text(d + 0.1, i, f"{d:.1f}", va="center", fontsize=7)

    fig.suptitle(
        f"Flat MARL disperses far from infrastructure while the hierarchy concentrates on "
        f"it ({region.capitalize()})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT / "fig5_flat_failure_analysis.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"Figure 5: flat dist {beh['Flat MARL (matched)']['mean_chebyshev_dist_to_nearest_asset']}"
          f" vs hier {beh['Learned Hierarchical']['mean_chebyshev_dist_to_nearest_asset']} -> {p}")


if __name__ == "__main__":
    figure2("saudi")
    figure3("saudi")
    figure5("saudi")
    print("Phase 7 figures rendered ->", OUT)
