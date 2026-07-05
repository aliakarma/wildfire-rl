"""Rollout rendering script to generate rollout GIFs for hierarchical policy and baselines.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import random
import numpy as np
import torch

from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.agents.strategic_controller import StrategicController, get_sector_center
from wildfire_marl.agents.agent_networks import MAPPOActor
from wildfire_marl.train.hierarchical_train import extract_high_level_state
from wildfire_marl.eval.metrics import burned_cells, infrastructure_survival_rate, weighted_economic_loss
from wildfire_marl.viz.rollout import render_rollout_frame, compile_rollout_gif
from scripts.run_multiseed_eval import run_episode_eval
from scripts.eval_hierarchical import target_seeking_action, get_greedy_risk_sectors, get_value_first_sectors

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def render_episode_rollout(
    env: MultiAgentFireEnv,
    setting: str,
    commander: StrategicController | None,
    flat_marl_actor: MAPPOActor | None,
    seed: int,
    device: torch.device,
    max_steps: int = 150,
) -> list[Any]:
    """Runs a single episode rollout, returning list of rendered PIL Image frames."""
    obs_dict, info_dict = env.reset(seed=seed)
    done = False
    step_count = 0
    high_level_interval = 10
    curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}
    env.apply_target_compliance = False
    
    frames = []
    
    while not done and step_count < max_steps:
        # High-level dispatch
        if step_count % high_level_interval == 0:
            if setting == "No-Op":
                curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}
            elif setting == "Greedy-Risk Heuristic":
                curr_targets = get_greedy_risk_sectors(env)
            elif setting == "Value-First Heuristic":
                curr_targets = get_value_first_sectors(env)
            elif setting == "Learned Hierarchical" and commander is not None:
                s_fire, s_asset, s_agent = extract_high_level_state(env)
                s_fire = s_fire.to(device)
                s_asset = s_asset.to(device)
                s_agent = s_agent.to(device)
                with torch.no_grad():
                    logits_list = commander(s_fire, s_asset, s_agent)
                    actions = [torch.argmax(l).item() for l in logits_list]
                for idx, agent in enumerate(env.agents):
                    curr_targets[agent] = get_sector_center(actions[idx])
            elif setting == "Flat MARL":
                curr_targets = {agent: env.agent_positions[agent] for agent in env.agents}
                
        env.strategic_targets = curr_targets
        
        # Low-level execution
        actions_dict = {}
        for agent in env.agents:
            mask = info_dict[agent]["action_mask"]
            if setting == "Flat MARL" and flat_marl_actor is not None:
                with torch.no_grad():
                    obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32, device=device).unsqueeze(0)
                    mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                    logits = flat_marl_actor(obs_t, mask_t).squeeze(0)
                    prob = torch.softmax(logits, dim=-1)
                    prob = prob * torch.tensor(mask, device=device).float()
                    if prob.sum() > 0:
                        actions_dict[agent] = int(torch.argmax(prob).item())
                    else:
                        actions_dict[agent] = 0
            elif setting == "No-Op":
                actions_dict[agent] = 0
            else:
                actions_dict[agent] = target_seeking_action(env, agent, mask)
                
        obs_dict, rewards_dict, terminations_dict, truncations_dict, info_dict = env.step(actions_dict)
        done = terminations_dict["agent_0"] or truncations_dict["agent_0"]
        
        # Compute step metrics for text overlays
        final_state = env.env.fire_state
        wel = weighted_economic_loss(
            asset_type=env.env.asset_type,
            final_state=final_state,
            asset_values=env.env.asset_values,
        )
        isr = infrastructure_survival_rate(
            asset_type=env.env.asset_type,
            final_state=final_state,
        )
        ce = info_dict[env.agents[0]].get("coordination_efficiency", 1.0)
        region_name = "saudi" if "saudi" in str(env.env.map_dir).lower() else "california"
        
        frame = render_rollout_frame(
            fire_state=final_state.copy(),
            asset_type=env.env.asset_type,
            agent_positions=env.agent_positions.copy(),
            strategic_targets=env.strategic_targets.copy(),
            step_idx=step_count,
            wel=wel,
            isr=isr,
            ce=ce,
            region=region_name,
            policy_name=setting
        )
        frames.append(frame)
        step_count += 1
        
    return frames

def main():
    parser = argparse.ArgumentParser(description="Render rollout GIFs.")
    parser.add_argument("--region", type=str, required=True, choices=["saudi", "california"])
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--output_gif", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    map_name = "Saudi" if args.region.lower() == "saudi" else "California"
    data_dir = "data/cell2fire"
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
    
    commander = None
    if args.policy == "Learned Hierarchical":
        hierarchical_ckpt = Path(f"results/runs/checkpoint_hierarchical_{args.region}.pt")
        if hierarchical_ckpt.exists():
            ckpt = torch.load(hierarchical_ckpt, map_location=device)
            commander = StrategicController(num_agents=env.num_agents).to(device)
            commander.load_state_dict(ckpt["commander_state_dict"])
            commander.eval()
            
    flat_marl_actor = None
    if args.policy == "Flat MARL":
        flat_ckpt = Path(f"results/runs/checkpoint_mappo_{args.region}.pt")
        if flat_ckpt.exists():
            ckpt = torch.load(flat_ckpt, map_location=device)
            flat_marl_actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
            flat_marl_actor.load_state_dict(ckpt["actor_state_dict"])
            flat_marl_actor.eval()
            
    print(f"Rendering rollout for {args.policy} on {args.region} ({args.steps} steps)...")
    frames = render_episode_rollout(
        env, args.policy, commander, flat_marl_actor, args.seed, device, max_steps=args.steps
    )
    env.close()
    
    if frames:
        compile_rollout_gif(frames, args.output_gif)
        print(f"Saved rollout animation to: {args.output_gif}")
    else:
        print("Error: No frames rendered.")

if __name__ == "__main__":
    main()
