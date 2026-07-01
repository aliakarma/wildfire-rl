"""Robustness testing utility to randomize environment parameters on reset."""

from __future__ import annotations

import numpy as np


def randomize_env_robustness(env: Any, rng: np.random.Generator) -> None:
    """Randomize env state channels and configurations to test robustness.

    Parameters randomized:
        1. **Ignition locations/counts**: Asymmetric multi-fire ignition (1 to 5 points).
        2. **Wind vectors**: Random wind direction angle and magnitude scaling.
        3. **Fuel densities**: Scale the fuel channel (channel 1) globally or locally.
        4. **Spread rates**: Random scaling of env base spread rate.
    """
    # 1. Randomize Ignition & Asymmetric multi-fire (channel 0)
    env.state[0] = 0.0
    h, w = env.state.shape[1], env.state.shape[2]
    n_points = rng.integers(2, 6)  # 2 to 5 multi-fire points
    for _ in range(n_points):
        rx = rng.integers(0, h)
        ry = rng.integers(0, w)
        env.state[0, rx, ry] = rng.uniform(0.7, 1.0)  # randomized starting intensity

    # 2. Randomize Wind vectors (channels 2 & 3)
    # Wind angle uniformly distributed in [0, 2*pi]
    angle = rng.uniform(0, 2 * np.pi)
    # Wind magnitude scaled randomly
    magnitude = rng.uniform(0.02, 0.15)
    env.state[2] = np.cos(angle) * magnitude
    env.state[3] = np.sin(angle) * magnitude

    # 3. Randomize Fuel densities (channel 1)
    # Scale existing fuel map globally by a factor of [0.7, 1.3]
    fuel_scale = rng.uniform(0.7, 1.3)
    env.state[1] = np.clip(env.state[1] * fuel_scale, 0.0, 1.0)

    # 4. Randomize Env Config Spread Coefficient
    # Slightly scale the base spread rate in config dynamically
    env.cfg.base_spread = env.cfg.base_spread * rng.uniform(0.8, 1.25)
