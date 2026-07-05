"""Hierarchical Two-Timescale training with Infrastructure-Weighted Commander Reward.

ARCHITECTURAL FIX:
- Low-level execution uses TARGET-SEEKING during both training AND eval.
- The frozen MAPPO actor was trained without target compliance, so it ignores commander
  targets completely. All strategic methods (Value-First, Greedy-Risk, Learned Hierarchical)
  use the same target-seeking low-level execution for a FAIR comparison.
- The commander's quality is measured purely by its sector selection decisions.

Commander reward = -(WEL_delta) + isr_weight * (ISR_delta)
This directly optimizes infrastructure protection, not noisy burned-area count.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from wildfire_marl.agents.agent_networks import MAPPOActor
from wildfire_marl.agents.strategic_controller import StrategicController, get_sector_center
from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.eval.metrics import infrastructure_survival_rate, weighted_economic_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_high_level_state(
    env: MultiAgentFireEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract sector-level fire, criticality, and agent position features."""
    fire = (env.env.fire_state > 0).astype(np.float32)
    crit = (
        env.env.criticality.astype(np.float32)
        if env.env.criticality is not None
        else np.zeros_like(fire)
    )

    sector_fire = np.zeros(16, dtype=np.float32)
    sector_asset = np.zeros(16, dtype=np.float32)

    for s in range(16):
        sy = s // 4
        sx = s % 4
        y_slice = slice(sy * 8, (sy + 1) * 8)
        x_slice = slice(sx * 8, (sx + 1) * 8)
        sector_fire[s] = float(np.sum(fire[y_slice, x_slice]))
        sector_asset[s] = float(np.sum(crit[y_slice, x_slice]))

    agent_sec = np.zeros(len(env.agents), dtype=np.float32)
    for i, agent in enumerate(env.agents):
        ay, ax = env.agent_positions[agent]
        sec_idx = (ay // 8) * 4 + (ax // 8)
        sec_idx = int(np.clip(sec_idx, 0, 15))
        agent_sec[i] = float(sec_idx)

    return (
        torch.tensor(sector_fire, dtype=torch.float32).unsqueeze(0),
        torch.tensor(sector_asset, dtype=torch.float32).unsqueeze(0),
        torch.tensor(agent_sec, dtype=torch.float32).unsqueeze(0),
    )


def target_seeking_action(env: MultiAgentFireEnv, agent: str, mask: list) -> int:
    """Move toward target sector, suppress fire locally when at target."""
    y, x = env.agent_positions[agent]
    ty, tx = env.strategic_targets[agent]
    dy = ty - y
    dx = tx - x

    # Move in the larger-magnitude direction first
    if abs(dy) >= abs(dx):
        if dy < 0 and (len(mask) > 1 and mask[1]):
            return 1  # up
        elif dy > 0 and (len(mask) > 2 and mask[2]):
            return 2  # down

    if dx < 0 and (len(mask) > 3 and mask[3]):
        return 3  # left
    elif dx > 0 and (len(mask) > 4 and mask[4]):
        return 4  # right

    # At target or blocked: suppress if possible
    if len(mask) > 5 and mask[5]:
        return 5  # suppress
    return 0  # noop


def get_infra_metrics(env: MultiAgentFireEnv) -> tuple[float, float]:
    """Compute current WEL and ISR from live fire state."""
    fire_state = env.env.fire_state
    asset_type = getattr(env.env, "asset_type", None)
    asset_values = getattr(env.env, "asset_values", {})

    wel = weighted_economic_loss(
        asset_type=asset_type,
        final_state=fire_state,
        asset_values=asset_values,
    )
    isr = infrastructure_survival_rate(
        asset_type=asset_type,
        final_state=fire_state,
    )
    return wel, isr


def get_value_first_sectors_action(env: MultiAgentFireEnv) -> list[int]:
    """Expert actions from Value-First Heuristic for 16 sectors."""
    s_fire, s_asset, _ = extract_high_level_state(env)
    s_fire_arr = s_fire.squeeze(0).numpy()
    s_asset_arr = s_asset.squeeze(0).numpy()

    risk_score = s_asset_arr * (s_fire_arr + 1.0)
    sorted_sectors = np.argsort(-risk_score)

    actions = []
    for idx in range(len(env.agents)):
        sec = sorted_sectors[idx % 16]
        actions.append(int(sec))
    return actions


def run_low_level_step(
    env: MultiAgentFireEnv,
    low_level_actor: MAPPOActor,
    obs_dict: dict,
    info_dict: dict,
    device: torch.device,
) -> tuple[dict, dict, dict, dict, dict]:
    """Run one step of frozen low-level agents."""
    actions_dict = {}
    for agent in env.agents:
        mask = info_dict[agent]["action_mask"]
        with torch.no_grad():
            obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            logits = low_level_actor(obs_t, mask_t).squeeze(0)
            prob = torch.softmax(logits, dim=-1)
            m = torch.tensor(mask, device=device)
            prob = prob * m.float()
            if prob.sum() > 0:
                actions_dict[agent] = int(torch.argmax(prob).item())
            else:
                actions_dict[agent] = 0

    return env.step(actions_dict)


def pretrain_commander(
    env: MultiAgentFireEnv,
    commander: StrategicController,
    num_episodes: int,
    bc_epochs: int,
    device: torch.device,
) -> None:
    """Pre-train commander via behavior cloning from Value-First Heuristic.

    Uses target-seeking low-level execution for consistency with RL fine-tuning.
    """
    print(f"Generating demonstrations for behavior cloning ({num_episodes} episodes)...")

    states_fire, states_asset, states_agent, expert_actions = [], [], [], []
    high_level_interval = 10
    env.apply_target_compliance = False

    for ep in range(num_episodes):
        obs_dict, info_dict = env.reset(seed=ep + 5000)
        done = False
        step_count = 0
        curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}

        while not done:
            if step_count % high_level_interval == 0:
                s_fire, s_asset, s_agent = extract_high_level_state(env)
                actions = get_value_first_sectors_action(env)

                states_fire.append(s_fire.squeeze(0).numpy())
                states_asset.append(s_asset.squeeze(0).numpy())
                states_agent.append(s_agent.squeeze(0).numpy())
                expert_actions.append(actions)

                for idx, agent in enumerate(env.agents):
                    curr_targets[agent] = get_sector_center(actions[idx])

            env.strategic_targets = curr_targets

            # Target-seeking low-level (same as RL fine-tuning)
            actions_dict = {
                agent: target_seeking_action(env, agent, info_dict[agent]["action_mask"])
                for agent in env.agents
            }
            obs_dict, _, terminations_dict, truncations_dict, info_dict = env.step(actions_dict)
            done = terminations_dict["agent_0"] or truncations_dict["agent_0"]
            step_count += 1

    print(f"Collected {len(expert_actions)} transitions. Training BC for {bc_epochs} epochs...")
    optimizer = optim.Adam(commander.parameters(), lr=1e-3)
    commander.train()

    t_fire = torch.tensor(np.array(states_fire), dtype=torch.float32, device=device)
    t_asset = torch.tensor(np.array(states_asset), dtype=torch.float32, device=device)
    t_agent = torch.tensor(np.array(states_agent), dtype=torch.float32, device=device)
    t_actions = torch.tensor(np.array(expert_actions), dtype=torch.long, device=device)

    for epoch in range(bc_epochs):
        logits_list = commander(t_fire, t_asset, t_agent)
        loss = torch.tensor(0.0, device=device)
        for i in range(len(env.agents)):
            loss = loss + F.cross_entropy(logits_list[i], t_actions[:, i])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"BC Epoch {epoch}/{bc_epochs} | Loss: {loss.item():.4f}")


def train_hierarchical(
    env: MultiAgentFireEnv,
    low_level_actor: MAPPOActor,  # kept for checkpoint compatibility; not used during RL fine-tuning
    cfg: dict[str, Any],
    device: torch.device,
) -> StrategicController:
    num_agents = env.num_agents
    pretrain_ep = int(cfg.get("pretrain_episodes", 40))
    bc_epochs = int(cfg.get("bc_epochs", 100))
    total_episodes = int(cfg.get("hierarchical_episodes", 120))
    high_level_interval = 10

    ent_coef = float(cfg.get("ent_coef", 0.05))
    kl_coef = float(cfg.get("kl_coef", 0.3))
    isr_weight = float(cfg.get("isr_weight", 20.0))
    gamma = float(cfg.get("gamma", 0.99))
    rl_lr = float(cfg.get("rl_lr", 5e-5))

    commander = StrategicController(num_agents=num_agents).to(device)

    # 1. BC pretraining (target-seeking low-level)
    if pretrain_ep > 0:
        pretrain_commander(env, commander, pretrain_ep, bc_epochs, device)

    # Frozen BC reference for KL regularization
    bc_reference = copy.deepcopy(commander)
    bc_reference.eval()
    for p in bc_reference.parameters():
        p.requires_grad_(False)

    # 2. RL Fine-tuning with infrastructure-specific commander reward
    optimizer = optim.Adam(commander.parameters(), lr=rl_lr)
    env.apply_target_compliance = False

    print(f"Starting hierarchical RL fine-tuning for {total_episodes} episodes...")
    print("  Low-level: target-seeking (fair comparison with all heuristics)")
    print(f"  Infra reward: -WEL_delta + {isr_weight}*ISR_delta")
    print(f"  KL coef: {kl_coef}, Entropy coef: {ent_coef}, LR: {rl_lr}")

    # EMA value baseline
    baseline = 0.0
    baseline_alpha = 0.1

    for ep in range(total_episodes):
        obs_dict, info_dict = env.reset()
        done = False
        step_count = 0

        states_buf_fire, states_buf_asset, states_buf_agent = [], [], []
        actions_buf, log_probs_buf, entropy_buf, kl_buf = [], [], [], []
        infra_rewards_buf = []

        curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}
        prev_wel, prev_isr = get_infra_metrics(env)
        pending_dispatch = False
        accumulated_raw_reward = 0.0

        while not done:
            if step_count % high_level_interval == 0:
                # Collect infrastructure reward for previous dispatch window
                if pending_dispatch:
                    if cfg.get("use_raw_reward", False):
                        infra_rew = accumulated_raw_reward
                    else:
                        curr_wel, curr_isr = get_infra_metrics(env)
                        well_delta = curr_wel - prev_wel
                        isr_delta = curr_isr - prev_isr
                        infra_rew = -well_delta + isr_weight * isr_delta
                    infra_rewards_buf.append(infra_rew)
                    accumulated_raw_reward = 0.0
                    if not cfg.get("use_raw_reward", False):
                        prev_wel, prev_isr = curr_wel, curr_isr
                else:
                    if not cfg.get("use_raw_reward", False):
                        prev_wel, prev_isr = get_infra_metrics(env)

                # Commander dispatch
                commander.train()
                s_fire, s_asset, s_agent = extract_high_level_state(env)
                s_fire = s_fire.to(device)
                s_asset = s_asset.to(device)
                s_agent = s_agent.to(device)

                logits_list = commander(s_fire, s_asset, s_agent)
                dists = [torch.distributions.Categorical(logits=logits) for logits in logits_list]
                actions = [d.sample() for d in dists]

                log_prob = sum([d.log_prob(a) for d, a in zip(dists, actions, strict=False)])
                entropy = sum([d.entropy() for d in dists])

                with torch.no_grad():
                    ref_logits_list = bc_reference(s_fire, s_asset, s_agent)
                    ref_dists = [
                        torch.distributions.Categorical(logits=logits) for logits in ref_logits_list
                    ]

                kl = sum(
                    [
                        torch.distributions.kl_divergence(d_curr, d_ref)
                        for d_curr, d_ref in zip(dists, ref_dists, strict=False)
                    ]
                )

                states_buf_fire.append(s_fire.cpu())
                states_buf_asset.append(s_asset.cpu())
                states_buf_agent.append(s_agent.cpu())
                actions_buf.append([a.item() for a in actions])
                log_probs_buf.append(log_prob)
                entropy_buf.append(entropy)
                kl_buf.append(kl)

                for idx, agent in enumerate(env.agents):
                    curr_targets[agent] = get_sector_center(int(actions[idx].item()))

                pending_dispatch = True

            env.strategic_targets = curr_targets

            # Tactical layer low-level selection (ablated vs standard)
            if cfg.get("use_mappo_tactical", False):
                next_obs, rewards_dict, terminations_dict, truncations_dict, next_info = (
                    run_low_level_step(env, low_level_actor, obs_dict, info_dict, device)
                )
            else:
                actions_dict = {
                    agent: target_seeking_action(env, agent, info_dict[agent]["action_mask"])
                    for agent in env.agents
                }
                next_obs, rewards_dict, terminations_dict, truncations_dict, next_info = env.step(
                    actions_dict
                )

            accumulated_raw_reward += rewards_dict["agent_0"]
            done = terminations_dict["agent_0"] or truncations_dict["agent_0"]
            obs_dict, info_dict = next_obs, next_info
            step_count += 1

        # Collect final infrastructure reward
        if pending_dispatch:
            if cfg.get("use_raw_reward", False):
                infra_rew = accumulated_raw_reward
            else:
                curr_wel, curr_isr = get_infra_metrics(env)
                infra_rew = -(curr_wel - prev_wel) + isr_weight * (curr_isr - prev_isr)
            infra_rewards_buf.append(infra_rew)

        # Align buffers
        n = min(len(log_probs_buf), len(infra_rewards_buf))
        if n == 0:
            continue

        log_probs_buf = log_probs_buf[:n]
        entropy_buf = entropy_buf[:n]
        kl_buf = kl_buf[:n]
        infra_rewards_buf = infra_rewards_buf[:n]

        # Discounted returns
        discounted_returns = []
        G = 0.0
        for r in reversed(infra_rewards_buf):
            G = r + gamma * G
            discounted_returns.insert(0, G)

        total_ep_infra_return = sum(infra_rewards_buf)
        baseline = (1 - baseline_alpha) * baseline + baseline_alpha * total_ep_infra_return

        # Policy Gradient update
        commander.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for idx in range(n):
            advantage = discounted_returns[idx] - baseline
            pg_loss = -log_probs_buf[idx] * advantage
            ent_loss = -ent_coef * entropy_buf[idx]
            kl_loss = kl_coef * kl_buf[idx]
            total_loss = total_loss + pg_loss + ent_loss + kl_loss

        total_loss.backward()
        nn.utils.clip_grad_norm_(commander.parameters(), max_norm=0.5)
        optimizer.step()

        if ep % 20 == 0:
            print(
                f"Episode {ep}/{total_episodes} | Infra Return: {total_ep_infra_return:.3f} | "
                f"Baseline: {baseline:.3f}"
            )

    env.apply_target_compliance = False
    return commander


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical strategic controller.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--set", type=str, action="append", default=[])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:
        key, val = kv.split("=", 1)
        try:
            cfg[key] = int(val)
        except ValueError:
            try:
                cfg[key] = float(val)
            except ValueError:
                cfg[key] = val

    set_seed(int(cfg.get("seed", 42)))

    region = cfg.get("region", "saudi")
    map_name = "Saudi" if region.lower() == "saudi" else "California"
    data_dir = cfg.get("data_dir", "data/cell2fire")
    infra_dir = f"{data_dir}/{map_name}"

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
        infra_dir=infra_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    low_level_ckpt = Path(f"results/runs/checkpoint_mappo_{region}.pt")
    if not low_level_ckpt.exists():
        raise FileNotFoundError(f"Missing MAPPO checkpoint: {low_level_ckpt}")

    low_level_actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
    ckpt = torch.load(low_level_ckpt, map_location=device)
    low_level_actor.load_state_dict(ckpt["actor_state_dict"])
    print(f"Loaded pre-trained low-level MAPPO Actor from {low_level_ckpt}")

    commander = train_hierarchical(env, low_level_actor, cfg, device)

    save_dir = Path("results/runs")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"checkpoint_hierarchical_{region}.pt"

    torch.save(
        {
            "commander_state_dict": commander.state_dict(),
            "actor_state_dict": low_level_actor.state_dict(),
            "config": cfg,
        },
        save_path,
    )
    print(f"Hierarchical strategic controller saved to {save_path}")

    env.close()


if __name__ == "__main__":
    main()
