"""Single-agent Maskable-PPO baseline implementation (Phase 4).

Utilizes stable-baselines3 and sb3-contrib to train and evaluate a CNN-based
reinforcement learning agent with action masking to prevent redundant/invalid actions.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

# Stable-baselines3 / sb3-contrib are lazy-imported inside functions
# to keep the package import light when torch/SB3 are not needed.


def make_custom_cnn_policy_kwargs(features_dim: int = 128) -> dict[str, Any]:
    """Build policy_kwargs targeting smaller grid dimensions (e.g. 32x32) without crashing."""
    import torch as th
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class CustomSmallCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
            super().__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # downsamples to H/2
                nn.ReLU(),
                nn.Flatten(),
            )
            # Compute input size for linear layer dynamically
            with th.no_grad():
                sample = th.as_tensor(observation_space.sample()[None]).float()
                n_flatten = self.cnn(sample).shape[1]

            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU()
            )

        def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.linear(self.cnn(observations))

    return dict(
        features_extractor_class=CustomSmallCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
    )


def wrap_env_for_maskable_ppo(env: gym.Env) -> gym.Env:
    """Wrap environment with ActionMasker for sb3_contrib."""
    from sb3_contrib.common.wrappers import ActionMasker

    def mask_fn(env_instance: gym.Env) -> np.ndarray:
        return env_instance.action_masks()

    return ActionMasker(env, mask_fn)


def train_ppo(
    region: str,
    total_timesteps: int = 50000,
    seed: int = 42,
    save_path: str | Path | None = None,
    observe_infra: bool = True,
    catastrophe_weight: float = 2.0,
    cascade_prob: float = 0.0,
) -> Any:
    """Train a Maskable-PPO model on the region landscape.

    Args:
        region: 'saudi' or 'california'.
        total_timesteps: Total timesteps to train.
        seed: Random seed for initialization.
        save_path: Path to save the trained model zip file.
        observe_infra: If True, include infrastructure channel in observations.
        catastrophe_weight: Economic cost weighting of asset damage.
        cascade_prob: Probability of cascade explosions.
    """
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    
    from wildfire_marl.env.single_agent_env import FireSuppressionEnv
    from wildfire_marl.env.rewards import InfrastructureWeightedReward, FireSizeReward
    
    # Map short name to capital directory name
    map_name = "Saudi" if region.lower() == "saudi" else "California"
    
    # Reward class
    reward_cls = InfrastructureWeightedReward if observe_infra else FireSizeReward
    
    print(f"Initializing training environment for region {region}...")
    raw_env = FireSuppressionEnv(
        fire_map=map_name,
        data_dir="data/cell2fire",
        steps_per_action=60,  # hourly steps
        reward_cls=reward_cls,
        observe_infra=observe_infra,
        catastrophe_weight=catastrophe_weight,
        cascade_prob=cascade_prob,
    )
    
    env = wrap_env_for_maskable_ppo(raw_env)
    
    policy_kwargs = make_custom_cnn_policy_kwargs(features_dim=128)
    
    print(f"Starting MaskablePPO training for {total_timesteps} steps (seed={seed})...")
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs,
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f"Model saved successfully to {save_path}")
        
    env.close()
    return model


class SB3PredictorWrapper:
    """Wraps an SB3 MaskablePPO model to present a standard Policy protocol."""

    def __init__(self, model: Any):
        self.model = model

    def predict(
        self,
        obs: np.ndarray,
        action_masks: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        # SB3 predict returns (action, state)
        action, state = self.model.predict(
            obs, action_masks=action_masks, deterministic=deterministic
        )
        # Handle single action output
        if isinstance(action, np.ndarray):
            action = int(action.item())
        return int(action), state


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/Evaluate PPO Baselines.")
    parser.add_argument("--train", action="store_true", help="Train a baseline PPO model.")
    parser.add_argument("--region", required=True, choices=["saudi", "california"], help="Region code.")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps to train.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--save-path", type=str, help="Save path for the model.")
    args = parser.parse_args()

    if args.train:
        save_path = args.save_path or f"models/baselines/ppo_{args.region}_seed_{args.seed}.zip"
        train_ppo(
            region=args.region,
            total_timesteps=args.timesteps,
            seed=args.seed,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
