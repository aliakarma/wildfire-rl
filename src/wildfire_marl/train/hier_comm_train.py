"""Training for the **Hierarchy + Communication** proposed model.

A value-aware strategic commander assigns each of N agents a high-value threatened asset every
``macro_interval`` steps; a *learned, communicating* tactical actor (:class:`CommTacticalActor`)
then defends its assigned asset — suppressing the fire frontier that protects it — conditioned on
the asset target and on mean-pooled messages from teammates. Unlike the original hierarchy —
whose deterministic Chebyshev target-seeking made the RL stage inert — the tactical layer is
trained end-to-end:

  1. **BC warm-start:** clone value-aware asset defense (:func:`asset_defense_action`), which
     already cuts WEL ~3x below the value-blind Local-Reactive baseline — a strong start.
  2. **PPO fine-tuning:** improve on the WEL/ISR objective (matched to evaluation) with a
     centralized critic and light firebreak shaping, so the team refines coordinated defense.

Commander modes:
  * ``heuristic`` — value-aware asset dispatch (:func:`_dispatch_value`); isolates the
    tactical-learning contribution.
  * ``learned``   — :class:`StrategicController`, BC-warmstarted then updated by REINFORCE with a
    KL anchor to the BC prior, jointly with the tactical PPO.

Shared parameters make everything agent-count agnostic (removes the paper's N=3 lock).
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

from wildfire_marl.agents.agent_networks import CommTacticalActor, MAPPOCritic
from wildfire_marl.agents.strategic_controller import StrategicController, get_sector_center
from wildfire_marl.env.regimes import make_marl_env
from wildfire_marl.env.rewards import WELISRDeltaReward
from wildfire_marl.eval.metrics import infrastructure_survival_rate, weighted_economic_loss
from wildfire_marl.train.hierarchical_train import (
    extract_high_level_state,
    get_value_first_sectors_action,
    pretrain_commander,
)

MACRO_INTERVAL = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _episode_wel_isr(env) -> tuple[float, float]:
    fs = env.env.fire_state
    return (
        float(weighted_economic_loss(env.env.asset_type, fs, env.env.asset_values)),
        float(infrastructure_survival_rate(env.env.asset_type, fs)),
    )


def asset_defense_action(env, agent: str, mask) -> int:
    """Strong tactical expert for BC: defend the assigned asset by suppressing the fire frontier
    *nearest that asset*, else advance toward it.

    The commander assigns each agent a high-value threatened asset (``env.strategic_targets``);
    this expert then treats the frontier cell between the fire and that asset, i.e. builds the
    firebreak that actually protects it. Cloning this *value-aware defense* — rather than
    value-blind nearest-fire suppression (Local Reactive) or walking to a sector center
    (target-seeking) — is what lets the learned tactical actor beat the reactive baseline: a
    scripted version of this policy cuts WEL ~3x below Local Reactive.
    """
    from scipy.ndimage import binary_dilation

    y, x = env.agent_positions[agent]
    ty0, tx0 = env.strategic_targets.get(agent, (y, x))
    fire = env.env.fire_state > 0
    if fire.any():
        frontier = binary_dilation(fire) & (env.env.fuel_mask > 0) & (env.env.fire_state == 0)
        fcand = np.argwhere(frontier)
        if len(fcand) > 0:
            d = (fcand[:, 0] - ty0) ** 2 + (fcand[:, 1] - tx0) ** 2
            ty, tx = fcand[int(np.argmin(d))]  # frontier cell defending the assigned asset
        else:
            ty, tx = ty0, tx0
    else:
        ty, tx = ty0, tx0

    dy, dx = ty - y, tx - x
    if abs(dy) + abs(dx) <= 1 and len(mask) > 5 and mask[5]:
        return 5  # adjacent to the defending frontier: treat
    if abs(dy) >= abs(dx):
        if dy < 0 and mask[1]:
            return 1
        if dy > 0 and mask[2]:
            return 2
    if dx < 0 and mask[3]:
        return 3
    if dx > 0 and mask[4]:
        return 4
    if len(mask) > 5 and mask[5]:
        return 5
    return 0


def compute_targets(env) -> np.ndarray:
    """Per-agent (rel_dy, rel_dx) toward the assigned target (asset/sector), in [-1, 1]. [N, 2]."""
    h, w = env.height, env.width
    out = np.zeros((env.num_agents, 2), dtype=np.float32)
    for i, agent in enumerate(env.agents):
        y, x = env.agent_positions[agent]
        ty, tx = env.strategic_targets.get(agent, (y, x))
        out[i, 0] = np.clip((ty - y) / h, -1.0, 1.0)
        out[i, 1] = np.clip((tx - x) / w, -1.0, 1.0)
    return out


def _dispatch_heuristic(env) -> None:
    """Value-First sector dispatch (writes env.strategic_targets)."""
    secs = get_value_first_sectors_action(env)
    for i, agent in enumerate(env.agents):
        env.strategic_targets[agent] = get_sector_center(secs[i])


def _dispatch_value(env) -> None:
    """Value-aware commander: assign each agent to a high-value THREATENED asset (asset value /
    distance-to-fire) and set its target to that asset's location. The learned tactical layer
    then defends the assigned asset via frontier suppression (:func:`asset_defense_action`)."""
    at = env.env.asset_type
    assets = np.argwhere(at > 0) if at is not None else np.empty((0, 2), dtype=int)
    if len(assets) == 0:
        _dispatch_heuristic(env)
        return
    vals = np.array([env.env.asset_values.get(int(at[y, x]), 1.0) for y, x in assets])
    fire_cells = np.argwhere(env.env.fire_state > 0)
    if len(fire_cells) > 0:
        threat = np.array(
            [
                vals[j]
                / (
                    1.0
                    + np.sqrt(np.min((fire_cells[:, 0] - ay) ** 2 + (fire_cells[:, 1] - ax) ** 2))
                )
                for j, (ay, ax) in enumerate(assets)
            ]
        )
        order = np.argsort(-threat)
    else:
        order = np.argsort(-vals)
    for i, agent in enumerate(env.agents):
        ay, ax = assets[order[i % len(assets)]]
        env.strategic_targets[agent] = (int(ay), int(ax))


def _dispatch_learned(env, commander, device) -> dict[str, Any]:
    """Sample sector assignments from the learned commander; return a macro transition."""
    s_fire, s_asset, s_agent = (t.to(device) for t in extract_high_level_state(env))
    logits_list = commander(s_fire, s_asset, s_agent)
    dists = [torch.distributions.Categorical(logits=lg) for lg in logits_list]
    acts = [d.sample() for d in dists]
    logp = sum(d.log_prob(a) for d, a in zip(dists, acts, strict=False))
    ent = sum(d.entropy() for d in dists)
    for i, agent in enumerate(env.agents):
        env.strategic_targets[agent] = get_sector_center(int(acts[i].item()))
    return {"state": (s_fire, s_asset, s_agent), "logp": logp, "ent": ent}


# ---------------------------------------------------------------------------- BC warm-start
def bc_warmstart_tactical(env, actor, episodes: int, epochs: int, device) -> None:
    """Behavior-clone the tactical actor to value-aware asset defense under the value commander."""
    print(f"[BC] Collecting tactical demonstrations ({episodes} episodes)...")
    env.apply_target_compliance = False
    obs_ts, tgt_ts, mask_ts, exp_ts = [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=ep + 7000)
        done, sc = False, 0
        while not done:
            if sc % MACRO_INTERVAL == 0:
                _dispatch_value(env)
            tgt = compute_targets(env)
            obs_ts.append([obs[a] for a in env.agents])
            tgt_ts.append(tgt)
            mask_ts.append([info[a]["action_mask"] for a in env.agents])
            exp_ts.append(
                [asset_defense_action(env, a, info[a]["action_mask"]) for a in env.agents]
            )
            acts = {a: exp_ts[-1][i] for i, a in enumerate(env.agents)}
            obs, _, term, trunc, info = env.step(acts)
            done = term["agent_0"] or trunc["agent_0"]
            sc += 1

    obs_t = torch.tensor(np.array(obs_ts), dtype=torch.float32, device=device)  # [T,N,8,K,K]
    tgt_t = torch.tensor(np.array(tgt_ts), dtype=torch.float32, device=device)  # [T,N,2]
    mask_t = torch.tensor(np.array(mask_ts), dtype=torch.bool, device=device)  # [T,N,6]
    exp_t = torch.tensor(np.array(exp_ts), dtype=torch.long, device=device)  # [T,N]
    print(f"[BC] Training tactical actor for {epochs} epochs on {obs_t.shape[0]} timesteps...")
    opt = optim.Adam(actor.parameters(), lr=1e-3)
    actor.train()
    bs = 256
    idx_all = np.arange(obs_t.shape[0])
    for epoch in range(epochs):
        np.random.shuffle(idx_all)
        tot = 0.0
        for s in range(0, len(idx_all), bs):
            bi = idx_all[s : s + bs]
            logits = actor(obs_t[bi], tgt_t[bi], mask_t[bi])  # [b,N,6]
            loss = F.cross_entropy(logits.reshape(-1, 6), exp_t[bi].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        if epoch % 15 == 0:
            print(f"[BC] epoch {epoch}/{epochs} loss {tot:.3f}")


# ---------------------------------------------------------------------------- PPO fine-tune
def train_hier_comm(env, cfg: dict[str, Any], device):
    n = env.num_agents
    comm_rounds = int(cfg.get("comm_rounds", 2))
    macro = int(cfg.get("macro_interval", MACRO_INTERVAL))
    commander_mode = str(cfg.get("commander", "heuristic"))

    actor = CommTacticalActor(
        in_channels=8, action_dim=6, features_dim=64, comm_rounds=comm_rounds
    ).to(device)
    critic = MAPPOCritic(in_channels=5, features_dim=128).to(device)

    commander = None
    bc_ref = None
    if commander_mode == "learned":
        commander = StrategicController(num_agents=n).to(device)
        if int(cfg.get("pretrain_episodes", 40)) > 0:
            pretrain_commander(
                env,
                commander,
                int(cfg.get("pretrain_episodes", 40)),
                int(cfg.get("bc_epochs", 100)),
                device,
            )
        bc_ref = copy.deepcopy(commander).eval()
        for p in bc_ref.parameters():
            p.requires_grad_(False)

    if int(cfg.get("bc_tactical_episodes", 20)) > 0:
        bc_warmstart_tactical(
            env,
            actor,
            int(cfg.get("bc_tactical_episodes", 20)),
            int(cfg.get("bc_tactical_epochs", 60)),
            device,
        )

    opt_a = optim.Adam(actor.parameters(), lr=float(cfg.get("lr_actor", 3e-4)))
    opt_c = optim.Adam(critic.parameters(), lr=float(cfg.get("lr_critic", 1e-3)))
    opt_cmd = (
        optim.Adam(commander.parameters(), lr=float(cfg.get("rl_lr", 5e-5)))
        if commander is not None
        else None
    )

    total_steps = int(cfg.get("total_steps", 50000))
    rollout_steps = int(cfg.get("n_steps", 1024))
    ppo_epochs = int(cfg.get("ppo_epochs", 4))
    batch_size = int(cfg.get("batch_size", 256))
    clip_eps = float(cfg.get("clip_eps", 0.2))
    gamma = float(cfg.get("gamma", 0.99))
    lam = float(cfg.get("gae_lambda", 0.95))
    ent_coef = float(cfg.get("ent_coef", 0.01))
    shaping_coef = float(cfg.get("shaping_coef", 0.0))
    kl_coef = float(cfg.get("kl_coef", 0.3))
    cmd_ent_coef = float(cfg.get("cmd_ent_coef", 0.05))

    env.apply_target_compliance = False
    step_count = 0
    ep_returns: list[float] = []
    train_curve: list[dict[str, float]] = []
    cmd_baseline = 0.0

    while step_count < total_steps:
        # --- collect a rollout of full episodes (~rollout_steps transitions) ---------------
        ep_obs, ep_tgt, ep_mask, ep_act, ep_logp, ep_val, ep_rew, ep_done, ep_state = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        macro_trans: list[dict[str, Any]] = []  # per learned-commander dispatch
        collected = 0
        actor.eval()
        critic.eval()

        while collected < rollout_steps:
            obs, info = env.reset()
            done, sc = False, 0
            ep_ret = 0.0
            burning_prev = int((env.env.fire_state > 0).sum())
            pending_macro: dict[str, Any] | None = None
            macro_r_accum = 0.0
            while not done:
                if sc % macro == 0:
                    if commander is not None:
                        if pending_macro is not None:
                            pending_macro["reward"] = macro_r_accum
                            macro_trans.append(pending_macro)
                            macro_r_accum = 0.0
                        pending_macro = _dispatch_learned(env, commander, device)
                    else:
                        _dispatch_value(env)

                obs_list = [obs[a] for a in env.agents]
                tgt = compute_targets(env)
                mask_list = [info[a]["action_mask"] for a in env.agents]
                obs_t = torch.tensor(
                    np.array(obs_list), dtype=torch.float32, device=device
                ).unsqueeze(0)
                tgt_t = torch.tensor(tgt, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.tensor(
                    np.array(mask_list), dtype=torch.bool, device=device
                ).unsqueeze(0)
                gstate = env.get_global_state()
                with torch.no_grad():
                    logits = actor(obs_t, tgt_t, mask_t)  # [1,N,6]
                    dist = torch.distributions.Categorical(logits=logits)
                    a_t = dist.sample()  # [1,N]
                    lp_t = dist.log_prob(a_t)  # [1,N]
                    v_t = critic(
                        torch.tensor(gstate, dtype=torch.float32, device=device).unsqueeze(0)
                    ).item()

                a_np = a_t.squeeze(0).cpu().numpy()
                acts = {a: int(a_np[i]) for i, a in enumerate(env.agents)}
                obs, rew_d, term_d, trunc_d, info = env.step(acts)
                r = float(rew_d["agent_0"])
                if shaping_coef != 0.0:
                    burning_now = int((env.env.fire_state > 0).sum())
                    r += shaping_coef * (burning_prev - burning_now)  # reward slowing spread
                    burning_prev = burning_now
                done = term_d["agent_0"] or trunc_d["agent_0"]

                ep_obs.append(obs_list)
                ep_tgt.append(tgt)
                ep_mask.append(mask_list)
                ep_act.append(a_np)
                ep_logp.append(lp_t.squeeze(0).cpu().numpy())
                ep_val.append(v_t)
                ep_rew.append(r)
                ep_done.append(done)
                ep_state.append(gstate)
                macro_r_accum += r
                ep_ret += r
                collected += 1
                sc += 1
                step_count += 1

            if commander is not None and pending_macro is not None:
                pending_macro["reward"] = macro_r_accum
                macro_trans.append(pending_macro)

            wel, isr = _episode_wel_isr(env)
            ep_returns.append(ep_ret)
            train_curve.append(
                {
                    "env_steps": step_count,
                    "episode": len(ep_returns),
                    "episode_return": ep_ret,
                    "WEL": wel,
                    "ISR": isr,
                }
            )

        # --- GAE (episodes delimited by done flags) ---------------------------------------
        actor.train()
        critic.train()
        vals = np.array(ep_val + [0.0], dtype=np.float32)
        rews = np.array(ep_rew, dtype=np.float32)
        dones = np.array(ep_done, dtype=np.float32)
        adv = np.zeros(len(ep_rew), dtype=np.float32)
        last = 0.0
        for t in reversed(range(len(ep_rew))):
            nonterm = 1.0 - dones[t]
            delta = rews[t] + gamma * vals[t + 1] * nonterm - vals[t]
            adv[t] = last = delta + gamma * lam * nonterm * last
        ret = adv + vals[:-1]

        obs_a = torch.tensor(np.array(ep_obs), dtype=torch.float32, device=device)  # [T,N,8,K,K]
        tgt_a = torch.tensor(np.array(ep_tgt), dtype=torch.float32, device=device)  # [T,N,2]
        mask_a = torch.tensor(np.array(ep_mask), dtype=torch.bool, device=device)  # [T,N,6]
        act_a = torch.tensor(np.array(ep_act), dtype=torch.long, device=device)  # [T,N]
        old_lp = torch.tensor(np.array(ep_logp), dtype=torch.float32, device=device)  # [T,N]
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device).unsqueeze(1)  # [T,1]
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device).unsqueeze(1)
        state_a = torch.tensor(
            np.array(ep_state), dtype=torch.float32, device=device
        )  # [T,5,32,32]
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        tt = obs_a.shape[0]
        for _ in range(ppo_epochs):
            perm = np.random.permutation(tt)
            for s in range(0, tt, batch_size):
                bi = perm[s : s + batch_size]
                logits = actor(obs_a[bi], tgt_a[bi], mask_a[bi])  # [b,N,6]
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(act_a[bi])  # [b,N]
                ent = dist.entropy().mean()
                ratio = torch.exp(new_lp - old_lp[bi])
                a_b = adv_t[bi]  # [b,1] broadcasts over agents
                surr1 = ratio * a_b
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a_b
                a_loss = -torch.min(surr1, surr2).mean() - ent_coef * ent
                opt_a.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                opt_a.step()

                v_pred = critic(state_a[bi]).squeeze(-1)
                c_loss = F.mse_loss(v_pred, ret_t[bi].squeeze(-1))
                opt_c.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt_c.step()

        # --- learned-commander policy-gradient update -------------------------------------
        if commander is not None and macro_trans:
            rewards = np.array([m["reward"] for m in macro_trans], dtype=np.float32)
            cmd_baseline = 0.9 * cmd_baseline + 0.1 * float(rewards.mean())
            opt_cmd.zero_grad()
            loss = torch.zeros((), device=device)
            for m in macro_trans:
                s_fire, s_asset, s_agent = m["state"]
                ref = bc_ref(s_fire, s_asset, s_agent)
                cur = commander(s_fire, s_asset, s_agent)
                kl = sum(
                    torch.distributions.kl_divergence(
                        torch.distributions.Categorical(logits=c),
                        torch.distributions.Categorical(logits=r),
                    )
                    for c, r in zip(cur, ref, strict=False)
                )
                adv_m = float(m["reward"]) - cmd_baseline
                loss = loss - m["logp"] * adv_m - cmd_ent_coef * m["ent"] + kl_coef * kl
            (loss / len(macro_trans)).backward()
            nn.utils.clip_grad_norm_(commander.parameters(), 0.5)
            opt_cmd.step()

        if ep_returns:
            print(
                f"Steps {step_count}/{total_steps} | ep_return(10) "
                f"{np.mean(ep_returns[-10:]):.2f} | last WEL {train_curve[-1]['WEL']:.1f} "
                f"ISR {train_curve[-1]['ISR']:.3f}"
            )

    return actor, critic, commander, train_curve


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Hierarchy+Comms proposed model.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--set", type=str, action="append", default=[])
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = int(v)
        except ValueError:
            try:
                cfg[k] = float(v)
            except ValueError:
                cfg[k] = v

    set_seed(int(cfg.get("seed", 42)))
    region = str(cfg.get("region", "saudi"))
    regime = str(cfg.get("regime", "default"))
    env = make_marl_env(
        region,
        regime=regime,
        data_dir=cfg.get("data_dir", "data/cell2fire"),
        num_agents=int(cfg.get("num_agents", 3)),
        reward_cls=WELISRDeltaReward,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Training HierComm | region={region} regime={regime} "
        f"commander={cfg.get('commander', 'heuristic')} device={device}"
    )

    actor, critic, commander, curve = train_hier_comm(env, cfg, device)

    save_dir = Path(cfg.get("save_dir", "results/runs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt: dict[str, Any] = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "config": cfg,
    }
    if commander is not None:
        ckpt["commander_state_dict"] = commander.state_dict()
    save_path = save_dir / f"checkpoint_hiercomm_{region}.pt"
    torch.save(ckpt, save_path)
    print(f"Saved HierComm checkpoint to {save_path}")

    if curve:
        import pandas as pd

        pd.DataFrame(curve).to_csv(save_dir / f"train_curve_hiercomm_{region}.csv", index=False)
    env.close()


if __name__ == "__main__":
    main()
