"""Training orchestrator for MAPPO and QMIX algorithms in the wildfire env.

Supports command line configuration parsing and saves checkpoints to results/runs/.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from wildfire_marl.agents.agent_networks import (
    MAPPOActor,
    MAPPOCritic,
    QMIXAgent,
    QMIXMixingNetwork,
)
from wildfire_marl.env.marl_env import MultiAgentFireEnv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


# --- Replay Buffer for QMIX ---------------------------------------------------
class QMIXReplayBuffer:
    def __init__(self, capacity: int = 5000, num_agents: int = 3, crop_size: int = 9):
        self.capacity = capacity
        self.num_agents = num_agents
        self.crop_size = crop_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, num_agents, 8, crop_size, crop_size), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, 8, crop_size, crop_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.states = np.zeros((capacity, 5, 32, 32), dtype=np.float32)
        self.next_states = np.zeros((capacity, 5, 32, 32), dtype=np.float32)

    def add(
        self,
        obs: dict[str, np.ndarray],
        actions: dict[str, int],
        reward: float,
        next_obs: dict[str, np.ndarray],
        done: bool,
        state: np.ndarray,
        next_state: np.ndarray,
    ):
        idx = self.ptr
        for i in range(self.num_agents):
            agent = f"agent_{i}"
            self.obs[idx, i] = obs[agent]
            self.actions[idx, i] = actions[agent]
            self.next_obs[idx, i] = next_obs[agent]

        self.rewards[idx] = reward
        self.dones[idx] = 1.0 if done else 0.0
        self.states[idx] = state
        self.next_states[idx] = next_state

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.tensor(self.obs[idxs], dtype=torch.float32),
            torch.tensor(self.actions[idxs], dtype=torch.long),
            torch.tensor(self.rewards[idxs], dtype=torch.float32),
            torch.tensor(self.next_obs[idxs], dtype=torch.float32),
            torch.tensor(self.dones[idxs], dtype=torch.float32),
            torch.tensor(self.states[idxs], dtype=torch.float32),
            torch.tensor(self.next_states[idxs], dtype=torch.float32),
        )


# --- MAPPO Trainer ------------------------------------------------------------
def train_mappo(
    env: MultiAgentFireEnv, cfg: dict[str, Any], device: torch.device
) -> tuple[MAPPOActor, MAPPOCritic]:
    num_agents = env.num_agents
    crop_size = env.crop_size

    actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
    critic = MAPPOCritic(in_channels=5, features_dim=128).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=float(cfg.get("lr_actor", 3e-4)))
    critic_optimizer = optim.Adam(critic.parameters(), lr=float(cfg.get("lr_critic", 1e-3)))

    total_steps = int(cfg.get("total_steps", 50000))
    n_steps = int(cfg.get("n_steps", 1024))
    batch_size = int(cfg.get("batch_size", 64))
    ppo_epochs = int(cfg.get("ppo_epochs", 4))
    clip_eps = float(cfg.get("clip_eps", 0.2))
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))

    obs_dict, info_dict = env.reset()
    global_state = env.get_global_state()

    step_count = 0
    ep_rewards = []
    curr_ep_reward = 0.0

    while step_count < total_steps:
        # Buffer to store rollouts
        obs_buf = []
        action_buf = []
        mask_buf = []
        reward_buf = []
        done_buf = []
        log_prob_buf = []
        val_buf = []
        state_buf = []

        actor.eval()
        critic.eval()

        for _ in range(n_steps):
            # Form tensor batch for all agents
            obs_list = [obs_dict[f"agent_{i}"] for i in range(num_agents)]
            mask_list = [info_dict[f"agent_{i}"]["action_mask"] for i in range(num_agents)]

            obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
            mask_t = torch.tensor(np.array(mask_list), dtype=torch.bool, device=device)

            with torch.no_grad():
                logits = actor(obs_t, mask_t)
                dist = torch.distributions.Categorical(logits=logits)
                actions_t = dist.sample()
                log_probs_t = dist.log_prob(actions_t)

                state_t = torch.tensor(global_state, dtype=torch.float32, device=device).unsqueeze(
                    0
                )
                val_t = critic(state_t).squeeze(0)  # Critic outputs [1, 1], squeezed to [1]

            actions_dict = {f"agent_{i}": int(actions_t[i].item()) for i in range(num_agents)}

            next_obs_dict, rewards_dict, terminations_dict, truncations_dict, next_info_dict = (
                env.step(actions_dict)
            )
            next_global_state = env.get_global_state()

            shared_reward = rewards_dict["agent_0"]
            done = terminations_dict["agent_0"] or truncations_dict["agent_0"]

            # Store in buffers
            obs_buf.append(obs_list)
            action_buf.append(actions_t.cpu().numpy())
            mask_buf.append(mask_list)
            reward_buf.append(shared_reward)
            done_buf.append(done)
            log_prob_buf.append(log_probs_t.cpu().numpy())
            val_buf.append(val_t.item())
            state_buf.append(global_state)

            curr_ep_reward += shared_reward
            step_count += 1

            if done:
                obs_dict, info_dict = env.reset()
                global_state = env.get_global_state()
                ep_rewards.append(curr_ep_reward)
                curr_ep_reward = 0.0
            else:
                obs_dict, info_dict = next_obs_dict, next_info_dict
                global_state = next_global_state

        # Compute GAE and returns
        actor.train()
        critic.train()

        with torch.no_grad():
            next_state_t = torch.tensor(global_state, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            next_val = critic(next_state_t).item()

        values = np.array(val_buf + [next_val])
        rewards = np.array(reward_buf)
        dones = np.array(done_buf)

        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * non_terminal * last_gae

        returns = advantages + values[:-1]

        # Convert buffers to tensors
        obs_arr = torch.tensor(
            np.array(obs_buf), dtype=torch.float32, device=device
        )  # [N_steps, N_agents, 8, K, K]
        act_arr = torch.tensor(
            np.array(action_buf), dtype=torch.long, device=device
        )  # [N_steps, N_agents]
        mask_arr = torch.tensor(
            np.array(mask_buf), dtype=torch.bool, device=device
        )  # [N_steps, N_agents, 6]
        old_log_probs = torch.tensor(
            np.array(log_prob_buf), dtype=torch.float32, device=device
        )  # [N_steps, N_agents]
        adv_arr = torch.tensor(advantages, dtype=torch.float32, device=device).unsqueeze(
            1
        )  # [N_steps, 1]
        ret_arr = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(
            1
        )  # [N_steps, 1]
        state_arr = torch.tensor(
            np.array(state_buf), dtype=torch.float32, device=device
        )  # [N_steps, 5, 32, 32]

        # Normalize advantages
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # Flatten timesteps and agent dimensions for Actor updates
        obs_flat = obs_arr.view(-1, 8, crop_size, crop_size)
        act_flat = act_arr.view(-1)
        mask_flat = mask_arr.view(-1, 6)
        old_log_probs_flat = old_log_probs.view(-1)
        adv_flat = adv_arr.repeat(1, num_agents).view(-1)

        # Optimize policy (PPO epochs)
        dataset_size = n_steps * num_agents
        for _ in range(ppo_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                # Actor update
                logits = actor(obs_flat[batch_idx], mask_flat[batch_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(act_flat[batch_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_flat[batch_idx])
                surr1 = ratio * adv_flat[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_flat[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                actor_optimizer.step()

            # Critic update (on global state)
            critic_indices = np.arange(n_steps)
            np.random.shuffle(critic_indices)
            for start in range(0, n_steps, batch_size // num_agents):
                end = start + batch_size // num_agents
                batch_idx = critic_indices[start:end]

                values_pred = critic(state_arr[batch_idx]).squeeze(-1)
                critic_loss = F.mse_loss(values_pred, ret_arr[batch_idx].squeeze(-1))

                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                critic_optimizer.step()

        # Logging progress
        if len(ep_rewards) > 0:
            mean_ep_rew = np.mean(ep_rewards[-10:])
            print(f"Steps: {step_count}/{total_steps} | Mean Return (10 eps): {mean_ep_rew:.2f}")

    return actor, critic


# --- QMIX Trainer -------------------------------------------------------------
def train_qmix(
    env: MultiAgentFireEnv, cfg: dict[str, Any], device: torch.device
) -> tuple[QMIXAgent, QMIXMixingNetwork]:
    num_agents = env.num_agents
    crop_size = env.crop_size

    agent_net = QMIXAgent(in_channels=8, action_dim=6, features_dim=64).to(device)
    target_agent_net = QMIXAgent(in_channels=8, action_dim=6, features_dim=64).to(device)
    target_agent_net.load_state_dict(agent_net.state_dict())

    mixer = QMIXMixingNetwork(num_agents=num_agents, state_dim=128, mix_hidden=32).to(device)
    target_mixer = QMIXMixingNetwork(num_agents=num_agents, state_dim=128, mix_hidden=32).to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    params = list(agent_net.parameters()) + list(mixer.parameters())
    optimizer = optim.Adam(params, lr=float(cfg.get("lr", 5e-4)))

    total_steps = int(cfg.get("total_steps", 50000))
    buffer = QMIXReplayBuffer(capacity=5000, num_agents=num_agents, crop_size=crop_size)

    batch_size = int(cfg.get("batch_size", 32))
    gamma = float(cfg.get("gamma", 0.99))
    target_update_interval = int(cfg.get("target_update_interval", 200))
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = int(total_steps * 0.6)

    obs_dict, info_dict = env.reset()
    global_state = env.get_global_state()

    step_count = 0
    ep_rewards = []
    curr_ep_reward = 0.0

    while step_count < total_steps:
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * step_count / epsilon_decay_steps,
        )

        actions_dict = {}
        for i in range(num_agents):
            agent = f"agent_{i}"
            mask = info_dict[agent]["action_mask"]
            if random.random() < epsilon:
                # Random valid action
                valid_actions = np.flatnonzero(mask)
                actions_dict[agent] = int(random.choice(valid_actions))
            else:
                agent_net.eval()
                with torch.no_grad():
                    obs_t = torch.tensor(
                        obs_dict[agent], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    q_vals = agent_net(obs_t).squeeze(0).cpu().numpy()
                # Apply mask (subtract large value from invalid actions)
                q_vals[~mask] = -1e9
                actions_dict[agent] = int(np.argmax(q_vals))

        next_obs_dict, rewards_dict, terminations_dict, truncations_dict, next_info_dict = env.step(
            actions_dict
        )
        next_global_state = env.get_global_state()

        shared_reward = rewards_dict["agent_0"]
        done = terminations_dict["agent_0"] or truncations_dict["agent_0"]

        buffer.add(
            obs_dict,
            actions_dict,
            shared_reward,
            next_obs_dict,
            done,
            global_state,
            next_global_state,
        )

        curr_ep_reward += shared_reward
        step_count += 1

        if done:
            obs_dict, info_dict = env.reset()
            global_state = env.get_global_state()
            ep_rewards.append(curr_ep_reward)
            curr_ep_reward = 0.0
        else:
            obs_dict, info_dict = next_obs_dict, next_info_dict
            global_state = next_global_state

        # Update weights if buffer has enough transitions
        if buffer.size >= batch_size and step_count % 4 == 0:
            agent_net.train()
            mixer.train()

            b_obs, b_actions, b_rewards, b_next_obs, b_dones, b_states, b_next_states = (
                buffer.sample(batch_size)
            )
            b_obs = b_obs.to(device)
            b_actions = b_actions.to(device)
            b_rewards = b_rewards.to(device)
            b_next_obs = b_next_obs.to(device)
            b_dones = b_dones.to(device)
            b_states = b_states.to(device)
            b_next_states = b_next_states.to(device)

            # 1. Calculate Q-values for current actions: Q_i(o_i, a_i)
            # b_obs shape: [batch, num_agents, 8, K, K]
            flat_obs = b_obs.view(-1, 8, crop_size, crop_size)
            flat_q = agent_net(flat_obs)  # [batch * num_agents, 6]
            qs = flat_q.view(batch_size, num_agents, 6)
            chosen_qs = torch.gather(qs, dim=2, index=b_actions.unsqueeze(-1)).squeeze(
                -1
            )  # [batch, num_agents]

            # 2. Calculate target max Q-values: max_a Q_i^-(o_i', a)
            flat_next_obs = b_next_obs.view(-1, 8, crop_size, crop_size)
            with torch.no_grad():
                flat_target_q = target_agent_net(flat_next_obs)
                target_qs = flat_target_q.view(batch_size, num_agents, 6)
                max_target_qs = torch.max(target_qs, dim=2)[0]  # [batch, num_agents]

            # 3. Calculate Q_tot and target Q_tot
            q_tot = mixer(chosen_qs, b_states)
            with torch.no_grad():
                target_q_tot = target_mixer(max_target_qs, b_next_states)
                y = b_rewards + gamma * target_q_tot * (1.0 - b_dones)

            # 4. TD loss
            loss = F.mse_loss(q_tot, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=0.5)
            optimizer.step()

            # Target networks soft/hard update
            if step_count % target_update_interval == 0:
                target_agent_net.load_state_dict(agent_net.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())

        # Progress printing
        if step_count % 2000 == 0 and len(ep_rewards) > 0:
            mean_ep_rew = np.mean(ep_rewards[-10:])
            print(
                f"Steps: {step_count}/{total_steps} | Mean Return: {mean_ep_rew:.2f} | Epsilon: {epsilon:.2f}"
            )

    return agent_net, mixer


def main():
    parser = argparse.ArgumentParser(description="Train wildfire multi-agent cooperative policies.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml file")
    parser.add_argument(
        "--set", type=str, default=None, help="Override parameter, e.g., total_steps=1000"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.set:
        key, val = args.set.split("=")
        cfg[key] = int(val) if val.isdigit() else val

    # Setup environment parameters
    region = cfg.get("region", "saudi")
    map_name = "Saudi" if region.lower() == "saudi" else "California"
    data_dir = cfg.get("data_dir", "data/cell2fire")
    infra_dir = f"{data_dir}/{map_name}"

    # Init environment
    env = MultiAgentFireEnv(
        num_agents=int(cfg.get("num_agents", 3)),
        crop_size=int(cfg.get("crop_size", 9)),
        coordination_penalty=float(cfg.get("coordination_penalty", 0.1)),
        fire_map=map_name,
        data_dir=data_dir,
        max_steps=int(cfg.get("max_steps", 150)),
        steps_per_action=60,  # hourly resolution
        observe_infra=True,
        catastrophe_weight=2.0,
        cascade_prob=0.1,
        infra_dir=infra_dir,
    )

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} | Region: {region.upper()}")

    algo = cfg.get("algo", "mappo").lower()
    save_dir = Path("results/runs")
    save_dir.mkdir(parents=True, exist_ok=True)

    if algo == "mappo":
        print("Starting MAPPO training...")
        actor, critic = train_mappo(env, cfg, device)
        save_path = save_dir / f"checkpoint_mappo_{region}.pt"
        torch.save(
            {
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "config": cfg,
            },
            save_path,
        )
        print(f"MAPPO policy saved successfully to {save_path}")
    elif algo == "qmix":
        print("Starting QMIX training...")
        agent_net, mixer = train_qmix(env, cfg, device)
        save_path = save_dir / f"checkpoint_qmix_{region}.pt"
        torch.save(
            {
                "agent_state_dict": agent_net.state_dict(),
                "mixer_state_dict": mixer.state_dict(),
                "config": cfg,
            },
            save_path,
        )
        print(f"QMIX policy saved successfully to {save_path}")
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    env.close()


if __name__ == "__main__":
    main()
