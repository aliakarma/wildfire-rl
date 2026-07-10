"""Unified Phase-3 policy evaluation on the suppression-relevant regime.

One rollout dispatcher for every policy class in the study — non-learned heuristics
(No-Op, Value-First, Greedy-Risk, Local-Reactive), the decentralized baselines
(MAPPO, QMIX, CommNet), and the proposed Hierarchy+Comms model (learned or heuristic
commander) — so all methods are scored under an identical protocol and env definition.

Checkpoints are loaded by convention from ``<ckpt_dir>/checkpoint_<kind>_<region>.pt``.
Evaluation uses ``reset(seed = seed_group + 100000 + episode)`` (disjoint from the training
ignition stream); the seed-group mean is the unit of analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from wildfire_marl.agents.agent_networks import (
    CommNetActor,
    CommTacticalActor,
    MAPPOActor,
    QMIXAgent,
)
from wildfire_marl.agents.strategic_controller import StrategicController, get_sector_center
from wildfire_marl.eval.metrics import (
    infrastructure_survival_rate,
    weighted_economic_loss,
)
from wildfire_marl.train.hier_comm_train import (
    _dispatch_heuristic,
    _dispatch_value,
    compute_targets,
)
from wildfire_marl.train.hierarchical_train import (
    extract_high_level_state,
    target_seeking_action,
)

MACRO_INTERVAL = 10
EVAL_OFFSET = 100_000

#: Human-readable labels used in tables.
LABELS: dict[str, str] = {
    "noop": "No-Op",
    "value_first": "Value-First",
    "greedy_risk": "Greedy-Risk",
    "local_reactive": "Local Reactive",
    "mappo": "Flat MARL (MAPPO)",
    "qmix": "QMIX",
    "commnet": "CommNet",
    "hiercomm": "HierComm (ours)",
    "hiercomm_heur": "HierComm (heur. cmd)",
}
#: Policy kinds that require a trained checkpoint.
LEARNED: set[str] = {"mappo", "qmix", "commnet", "hiercomm", "hiercomm_heur"}
ALL_KINDS: list[str] = list(LABELS)


def checkpoint_path(kind: str, region: str, ckpt_dir: str | Path, seed: int | None = None) -> Path:
    """Checkpoint file a learned method saves/loads.

    ``seed`` appends an ``_s<seed>`` suffix for the multi-training-seed sweep; omit it for the
    single-seed convention (backward compatible). ``hiercomm_heur`` shares the heuristic-commander
    run's stem.
    """
    stem = "hiercomm_heur" if kind == "hiercomm_heur" else kind
    suffix = f"_s{seed}" if seed is not None else ""
    return Path(ckpt_dir) / f"checkpoint_{stem}_{region}{suffix}.pt"


def load_nets(
    kind: str, region: str, ckpt_dir: str | Path, num_agents: int, device, seed: int | None = None
) -> dict[str, Any]:
    """Load the network(s) for a learned policy; returns {} for non-learned heuristics."""
    if kind not in LEARNED:
        return {}
    path = checkpoint_path(kind, region, ckpt_dir, seed=seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for '{kind}' ({region}): {path}")
    ckpt = torch.load(path, map_location=device)
    nets: dict[str, Any] = {}
    if kind == "mappo":
        net = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
        net.load_state_dict(ckpt["actor_state_dict"])
        net.eval()
        nets["actor"] = net
    elif kind == "commnet":
        net = CommNetActor(in_channels=8, action_dim=6, features_dim=64).to(device)
        net.load_state_dict(ckpt["actor_state_dict"])
        net.eval()
        nets["actor"] = net
    elif kind == "qmix":
        net = QMIXAgent(in_channels=8, action_dim=6, features_dim=64).to(device)
        net.load_state_dict(ckpt["agent_state_dict"])
        net.eval()
        nets["agent"] = net
    else:  # hiercomm / hiercomm_heur
        actor = CommTacticalActor(
            in_channels=8,
            action_dim=6,
            features_dim=64,
            comm_rounds=int(ckpt.get("config", {}).get("comm_rounds", 2)),
        ).to(device)
        actor.load_state_dict(ckpt["actor_state_dict"])
        actor.eval()
        nets["actor"] = actor
        if "commander_state_dict" in ckpt:
            cmd = StrategicController(num_agents=num_agents).to(device)
            cmd.load_state_dict(ckpt["commander_state_dict"])
            cmd.eval()
            nets["commander"] = cmd
    return nets


def _greedy_risk_dispatch(env) -> None:
    """Dispatch each agent to a top fire-load sector."""
    s_fire, _, _ = extract_high_level_state(env)
    order = np.argsort(-s_fire.squeeze(0).numpy())
    for i, a in enumerate(env.agents):
        env.strategic_targets[a] = get_sector_center(int(order[i % 16]))


def _reactive_targets(env) -> None:
    """Local-Reactive: send each agent to the nearest treatable fire-frontier cell (broad
    coverage), falling back to the nearest asset when no fire is present."""
    from scipy.ndimage import binary_dilation

    fire = env.env.fire_state > 0
    fuel = env.env.fuel_mask > 0
    clean = env.env.fire_state == 0
    if fire.any():
        cand = np.argwhere(binary_dilation(fire) & fuel & clean)
        if len(cand) == 0:
            cand = np.argwhere(fuel & clean)
    elif env.env.asset_type is not None and (env.env.asset_type > 0).any():
        cand = np.argwhere(env.env.asset_type > 0)
    else:
        cand = np.argwhere(fuel)
    for a in env.agents:
        y, x = env.agent_positions[a]
        d = (cand[:, 0] - y) ** 2 + (cand[:, 1] - x) ** 2
        ty, tx = cand[int(np.argmin(d))]
        env.strategic_targets[a] = (int(ty), int(tx))


def _flat_actions(env, net, obs, info, device, comms: bool) -> dict[str, int]:
    """Greedy masked actions for a decentralized actor (MAPPO/CommNet)."""
    masks = [info[a]["action_mask"] for a in env.agents]
    obs_arr = np.array([obs[a] for a in env.agents])
    mask_t = torch.tensor(np.array(masks), dtype=torch.bool, device=device)
    with torch.no_grad():
        if comms:
            logits = net(
                torch.tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0),
                mask_t.unsqueeze(0),
            ).squeeze(0)
        else:
            logits = net(torch.tensor(obs_arr, dtype=torch.float32, device=device), mask_t)
    return {a: int(logits[i].argmax().item()) for i, a in enumerate(env.agents)}


def _qmix_actions(env, net, obs, info, device) -> dict[str, int]:
    acts = {}
    for a in env.agents:
        mask = np.asarray(info[a]["action_mask"])
        with torch.no_grad():
            q = net(torch.tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0)).squeeze(
                0
            )
        q = q.cpu().numpy()
        q[~mask] = -1e9
        acts[a] = int(np.argmax(q))
    return acts


def rollout_episode(env, kind: str, nets: dict[str, Any], device, seed: int) -> dict[str, float]:
    """Run one evaluation episode; return WEL / ISR / CE / burned / return."""
    obs, info = env.reset(seed=seed)
    env.apply_target_compliance = False
    done, sc, ep_ret = False, 0, 0.0
    ce = 1.0
    while not done:
        # strategic dispatch (every macro interval) for hierarchical / heuristic policies
        if sc % MACRO_INTERVAL == 0:
            if kind == "value_first":
                _dispatch_heuristic(env)
            elif kind == "greedy_risk":
                _greedy_risk_dispatch(env)
            elif kind == "hiercomm" and "commander" in nets:
                s = [t.to(device) for t in extract_high_level_state(env)]
                with torch.no_grad():
                    logits_list = nets["commander"](*s)
                for i, a in enumerate(env.agents):
                    env.strategic_targets[a] = get_sector_center(
                        int(logits_list[i].argmax().item())
                    )
            elif kind in ("hiercomm_heur", "hiercomm"):
                # value-aware asset dispatch (heuristic commander / learned-tactical fallback)
                _dispatch_value(env)
        if kind == "local_reactive":
            _reactive_targets(env)

        # low-level actions
        if kind == "noop":
            acts = dict.fromkeys(env.agents, 0)
        elif kind in ("value_first", "greedy_risk", "local_reactive"):
            acts = {a: target_seeking_action(env, a, info[a]["action_mask"]) for a in env.agents}
        elif kind == "mappo":
            acts = _flat_actions(env, nets["actor"], obs, info, device, comms=False)
        elif kind == "commnet":
            acts = _flat_actions(env, nets["actor"], obs, info, device, comms=True)
        elif kind == "qmix":
            acts = _qmix_actions(env, nets["agent"], obs, info, device)
        else:  # hiercomm / hiercomm_heur -> learned comms-tactical actor
            tgt = compute_targets(env)
            obs_t = torch.tensor(
                np.array([obs[a] for a in env.agents]), dtype=torch.float32, device=device
            ).unsqueeze(0)
            tgt_t = torch.tensor(tgt, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(
                np.array([info[a]["action_mask"] for a in env.agents]),
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                logits = nets["actor"](obs_t, tgt_t, mask_t).squeeze(0)
            acts = {a: int(logits[i].argmax().item()) for i, a in enumerate(env.agents)}

        obs, rew, term, trunc, info = env.step(acts)
        ep_ret += float(rew["agent_0"])
        done = term["agent_0"] or trunc["agent_0"]
        ce = info[env.agents[0]].get("coordination_efficiency", ce)
        sc += 1

    fs = env.env.fire_state
    return {
        "WEL": float(weighted_economic_loss(env.env.asset_type, fs, env.env.asset_values)),
        "ISR": float(infrastructure_survival_rate(env.env.asset_type, fs)),
        "CE": float(ce),
        "burned": int((fs > 0).sum()),
        "return": ep_ret,
    }


def evaluate(
    env,
    kind: str,
    region: str,
    seeds: list[int],
    episodes: int,
    ckpt_dir: str | Path,
    device,
    train_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate ``kind`` over ``seeds`` x ``episodes``; return per-episode records.

    ``train_seed`` selects a specific training-seed checkpoint (``_s<seed>``) for the multi-seed
    sweep; ``None`` uses the single-seed checkpoint.
    """
    nets = load_nets(kind, region, ckpt_dir, env.num_agents, device, seed=train_seed)
    records = []
    for s in seeds:
        for e in range(episodes):
            m = rollout_episode(env, kind, nets, device, seed=s + EVAL_OFFSET + e)
            m.update({"region": region, "policy": kind, "seed": s, "episode": e})
            records.append(m)
    return records
