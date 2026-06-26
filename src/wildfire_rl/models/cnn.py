"""Single canonical CNN feature extractor (replaces ~7 copy-pasted ``CustomCNN``).

Adds an optional pooling stack (``use_pooling=True``). The original network had no
spatial downsampling, so on a 32x32 grid the first linear layer was Linear(65536, 256)
(~16.7M params) — the main reason each PPO checkpoint was ~200 MB. With pooling the
flattened dimension drops ~16x, producing far smaller, git/HF-friendly checkpoints.
"""

from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CustomCNN(BaseFeaturesExtractor):
    """Conv stack -> flatten -> linear projection, for ``(C, H, W)`` wildfire tensors."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        use_pooling: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        layers: list[nn.Module] = [
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ]
        if use_pooling:
            layers.append(nn.MaxPool2d(2))  # H/2
        layers += [
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ]
        if use_pooling:
            layers.append(nn.MaxPool2d(2))  # H/4
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
