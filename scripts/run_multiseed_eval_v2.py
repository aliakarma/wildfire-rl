"""Phase 2 (peer-review remediation) multi-seed evaluation.

Evaluates the rehabilitated baselines (MAPPO / QMIX / CommNet trained for 100k steps on the
WEL/ISR-matched objective, results/phase2_peer/) alongside the original policies under the
certified protocol (5 seeds x 15 episodes, identical seed derivation to phase8), with
behavioral instrumentation: per-policy action histograms, occupancy maps, and mean distance
to the nearest asset (peer-review issues H1, H3, M3, M7).

The evaluation environment is byte-identical to the certified phase8 configuration
(including the default FireSizeReward for the reported total_reward column) so WEL/ISR/CE
are directly comparable with Table 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps
from scripts.eval_hierarchical import (
    get_greedy_risk_sectors,
    get_value_first_sectors,
    target_seeking_action,
)

from wildfire_marl.agents.agent_networks import CommNetActor, MAPPOActor, QMIXAgent
from wildfire_marl.agents.strategic_controller import StrategicController, get_sector_center
from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.eval.metrics import (
    burned_cells,
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.train.hierarchical_train import extract_high_level_state


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BehaviorLog:
    """Accumulates action histogram, occupancy map, and asset-distance stats per policy."""

    def __init__(self, height: int, width: int, asset_type: np.ndarray | None):
        self.action_counts = np.zeros(6, dtype=np.int64)
        self.occupancy = np.zeros((height, width), dtype=np.int64)
        self.dist_sum = 0.0
        self.dist_n = 0
        if asset_type is not None and (asset_type > 0).any():
            ys, xs = np.nonzero(asset_type > 0)
            self.asset_coords = np.stack([ys, xs], axis=1)
        else:
            self.asset_coords = None

    def record(self, actions: dict[str, int], positions: dict[str, tuple[int, int]]):
        for agent, act in actions.items():
            self.action_counts[int(act)] += 1
            y, x = positions[agent]
            self.occupancy[y, x] += 1
            if self.asset_coords is not None:
                d = np.abs(self.asset_coords - np.array([y, x])).max(axis=1).min()
                self.dist_sum += float(d)
                self.dist_n += 1

    def summary(self) -> dict:
        total = int(self.action_counts.sum())
        return {
            "action_counts": self.action_counts.tolist(),
            "action_fractions": (self.action_counts / max(total, 1)).round(4).tolist(),
            "mean_chebyshev_dist_to_nearest_asset": (
                round(self.dist_sum / self.dist_n, 3) if self.dist_n else None
            ),
        }


def run_episode(
    env: MultiAgentFireEnv,
    setting: str,
    nets: dict,
    seed: int,
    device: torch.device,
    behavior: BehaviorLog,
    reset_options: dict | None = None,
) -> dict[str, float]:
    obs_dict, info_dict = env.reset(seed=seed, options=reset_options)
    env.apply_target_compliance = False

    curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}
    done = False
    step_count = 0
    high_level_interval = 10
    ep_reward = 0.0

    while not done:
        if step_count % high_level_interval == 0:
            if setting == "Greedy-Risk Heuristic":
                curr_targets = get_greedy_risk_sectors(env)
            elif setting == "Value-First Heuristic":
                curr_targets = get_value_first_sectors(env)
            elif setting == "Learned Hierarchical":
                s_fire, s_asset, s_agent = extract_high_level_state(env)
                with torch.no_grad():
                    logits_list = nets["commander"](
                        s_fire.to(device), s_asset.to(device), s_agent.to(device)
                    )
                    actions = [torch.argmax(logits).item() for logits in logits_list]
                for idx, agent in enumerate(env.agents):
                    curr_targets[agent] = get_sector_center(actions[idx])
            else:
                # No-Op and flat learned policies: target = own position (as certified)
                curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}

        env.strategic_targets = curr_targets

        actions_dict: dict[str, int] = {}
        if setting == "CommNet (matched)":
            obs_list = [obs_dict[a] for a in env.agents]
            mask_list = [info_dict[a]["action_mask"] for a in env.agents]
            obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
            mask_t = torch.tensor(np.array(mask_list), dtype=torch.bool, device=device)
            with torch.no_grad():
                logits = nets["commnet"](obs_t.unsqueeze(0), mask_t.unsqueeze(0)).squeeze(0)
            acts = torch.argmax(logits, dim=-1).cpu().numpy()
            actions_dict = {a: int(acts[i]) for i, a in enumerate(env.agents)}
        else:
            for agent in env.agents:
                mask = info_dict[agent]["action_mask"]
                if setting == "No-Op":
                    actions_dict[agent] = 0
                elif setting == "Local Reactive":
                    # Treat-if-possible, else random valid move (the unreported
                    # "Heuristic" of eval_marl.py — audit issue M11). Uses the env's
                    # seeded RNG for reproducibility.
                    if mask[5]:
                        actions_dict[agent] = 5
                    else:
                        valid_moves = np.flatnonzero(mask[:5])
                        actions_dict[agent] = int(env.env.np_random.choice(valid_moves))
                elif setting == "Flat MARL (matched)":
                    with torch.no_grad():
                        obs_t = torch.tensor(
                            obs_dict[agent], dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        mask_t = torch.tensor(
                            mask, dtype=torch.bool, device=device
                        ).unsqueeze(0)
                        logits = nets["mappo"](obs_t, mask_t).squeeze(0)
                    actions_dict[agent] = int(torch.argmax(logits).item())
                elif setting == "QMIX (matched)":
                    with torch.no_grad():
                        obs_t = torch.tensor(
                            obs_dict[agent], dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        q = nets["qmix"](obs_t).squeeze(0).cpu().numpy()
                    q[~mask] = -1e9
                    actions_dict[agent] = int(np.argmax(q))
                else:  # heuristics + hierarchical: deterministic target seeking
                    actions_dict[agent] = target_seeking_action(env, agent, mask)

        obs_dict, rewards_dict, terminations_dict, truncations_dict, info_dict = env.step(
            actions_dict
        )
        ep_reward += rewards_dict["agent_0"]
        done = terminations_dict["agent_0"] or truncations_dict["agent_0"]

        behavior.record(actions_dict, env.agent_positions)
        step_count += 1

    final_state = env.env.fire_state
    return {
        "WEL": weighted_economic_loss(
            asset_type=env.env.asset_type,
            final_state=final_state,
            asset_values=env.env.asset_values,
        ),
        "ISR": infrastructure_survival_rate(
            asset_type=env.env.asset_type, final_state=final_state
        ),
        "CE": info_dict[env.agents[0]].get("coordination_efficiency", 1.0),
        "burned_cells": burned_cells(final_state),
        "total_reward": ep_reward,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 100 * alpha / 2)), float(
        np.percentile(boots, 100 * (1 - alpha / 2))
    )


def welch_and_d(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch's t-test and pooled-SD Cohen's d of a vs b (positive = a larger)."""
    if np.allclose(a.std(), 0) and np.allclose(b.std(), 0):
        return {"t_stat": None, "p_val": None, "cohens_d": None, "note": "both degenerate"}
    t, p = sps.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else None
    return {"t_stat": float(t), "p_val": float(p), "cohens_d": d}


def main():
    parser = argparse.ArgumentParser(description="Phase 2 remediation multi-seed evaluation.")
    parser.add_argument("--region", type=str, required=True, choices=["saudi", "california"])
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default="results/phase2_peer")
    parser.add_argument("--output_dir", type=str, default="results/phase2_peer")
    parser.add_argument(
        "--policies",
        type=str,
        default="all",
        help='comma-separated subset (e.g. "Local Reactive"), or "all"',
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="output filename suffix (e.g. _extra)"
    )
    args = parser.parse_args()

    set_seed(args.base_seed)
    device = torch.device("cpu")

    map_name = "Saudi" if args.region.lower() == "saudi" else "California"
    data_dir = "data/cell2fire"
    env = MultiAgentFireEnv(
        num_agents=3,
        crop_size=9,
        coordination_penalty=0.1,
        fire_map=map_name,
        data_dir=data_dir,
        max_steps=150,
        steps_per_action=60,
        observe_infra=True,
        catastrophe_weight=2.0,
        cascade_prob=0.1,
        infra_dir=f"{data_dir}/{map_name}",
    )

    nets: dict = {}
    ckpt_dir = Path(args.ckpt_dir)

    ck = torch.load(
        Path(f"results/runs/checkpoint_hierarchical_{args.region}.pt"), map_location=device
    )
    commander = StrategicController(num_agents=env.num_agents).to(device)
    commander.load_state_dict(ck["commander_state_dict"])
    commander.eval()
    nets["commander"] = commander

    ck = torch.load(ckpt_dir / f"checkpoint_mappo_{args.region}.pt", map_location=device)
    mappo = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
    mappo.load_state_dict(ck["actor_state_dict"])
    mappo.eval()
    nets["mappo"] = mappo

    ck = torch.load(ckpt_dir / f"checkpoint_qmix_{args.region}.pt", map_location=device)
    qmix = QMIXAgent(in_channels=8, action_dim=6, features_dim=64).to(device)
    qmix.load_state_dict(ck["agent_state_dict"])
    qmix.eval()
    nets["qmix"] = qmix

    ck = torch.load(ckpt_dir / f"checkpoint_commnet_{args.region}.pt", map_location=device)
    commnet = CommNetActor(in_channels=8, action_dim=6, features_dim=64, comm_rounds=2).to(
        device
    )
    commnet.load_state_dict(ck["actor_state_dict"])
    commnet.eval()
    nets["commnet"] = commnet

    policies = [
        "No-Op",
        "Greedy-Risk Heuristic",
        "Value-First Heuristic",
        "Local Reactive",
        "Flat MARL (matched)",
        "QMIX (matched)",
        "CommNet (matched)",
        "Learned Hierarchical",
    ]
    if args.policies != "all":
        requested = [p.strip() for p in args.policies.split(",")]
        unknown = set(requested) - set(policies)
        if unknown:
            raise ValueError(f"Unknown policies: {unknown}")
        policies = requested

    eval_seeds = [args.base_seed + s * 1000 for s in range(args.seeds)]
    records = []
    behaviors = {
        p: BehaviorLog(env.height, env.width, env.env.asset_type) for p in policies
    }

    for seed in eval_seeds:
        for policy in policies:
            print(f"[{args.region}] {policy} | seed {seed} | {args.episodes} episodes",
                  flush=True)
            for ep in range(args.episodes):
                metrics = run_episode(
                    env, policy, nets, seed + ep, device, behaviors[policy]
                )
                records.append(
                    {"seed": seed, "episode": ep, "region": args.region, "policy": policy}
                    | metrics
                )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sfx = args.suffix
    raw = pd.DataFrame(records)
    raw.to_csv(out / f"multiseed_eval_raw_{args.region}{sfx}.csv", index=False)

    agg = raw.groupby(["region", "policy", "seed"], as_index=False).mean(numeric_only=True)
    agg.to_csv(out / f"multiseed_eval_aggregate_{args.region}{sfx}.csv", index=False)

    # Summary stats over 5 seed means (statistical unit = seed, n = 5)
    summary: dict = {"region": args.region, "protocol": {
        "seeds": eval_seeds, "episodes_per_seed": args.episodes, "unit": "seed means"
    }, "metrics": {}, "comparisons": {}}
    for metric in ["WEL", "ISR", "CE", "burned_cells", "total_reward"]:
        summary["metrics"][metric] = {}
        for policy in policies:
            v = agg.loc[agg.policy == policy, metric].to_numpy()
            lo, hi = bootstrap_ci(v)
            summary["metrics"][metric][policy] = {
                "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "ci_lo": lo, "ci_hi": hi, "per_seed": [round(float(x), 4) for x in v],
            }
    if "No-Op" in policies and "Learned Hierarchical" in policies:
        for metric in ["WEL", "ISR"]:
            summary["comparisons"][metric] = {}
            hier = agg.loc[agg.policy == "Learned Hierarchical", metric].to_numpy()
            noop = agg.loc[agg.policy == "No-Op", metric].to_numpy()
            for policy in policies:
                if policy == "No-Op":
                    continue
                v = agg.loc[agg.policy == policy, metric].to_numpy()
                summary["comparisons"][metric][policy] = {
                    "vs_NoOp": welch_and_d(v, noop),
                    "vs_Hierarchical": welch_and_d(v, hier),
                }

    with open(out / f"eval_summary_{args.region}{sfx}.json", "w") as f:
        json.dump(summary, f, indent=2)

    behavior_out = {p: behaviors[p].summary() for p in policies}
    with open(out / f"behavior_{args.region}{sfx}.json", "w") as f:
        json.dump(behavior_out, f, indent=2)
    np.savez_compressed(
        out / f"occupancy_{args.region}{sfx}.npz",
        **{p.replace(" ", "_"): behaviors[p].occupancy for p in policies},
    )

    env.close()
    print(f"[{args.region}] evaluation complete -> {out}")


if __name__ == "__main__":
    main()
