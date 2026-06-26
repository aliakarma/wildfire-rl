"""Configuration schema and loader (OmegaConf structured configs).

This replaces every hardcoded constant and path in the notebooks. Configs are plain
YAML under ``configs/`` and validated against the dataclasses below. Dotlist overrides
(e.g. ``ppo.total_timesteps=1000``) make CLI/CI parameterization trivial and are fully
Hydra-compatible if a Hydra app is added later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

# Canonical 7-channel ordering — preserved from the original tensors and frozen here
# so every region/environment agrees (channels 0..6).
DEFAULT_CHANNELS: list[str] = [
    "fire",
    "fuel",
    "wind_x",
    "wind_y",
    "terrain",
    "temperature",
    "humidity",
]


@dataclass
class EnvConfig:
    """Wildfire environment dynamics. Every magic number from the notebooks lives here."""

    grid_size: int = 32
    max_steps: int = 200
    n_channels: int = 7
    channels: list[str] = field(default_factory=lambda: list(DEFAULT_CHANNELS))

    # --- fire-spread model: spread_prob = base + fuel*f + wind*w + terrain*t ---
    base_spread: float = 0.01
    fuel_coeff: float = 0.15
    wind_coeff: float = 0.08
    terrain_coeff: float = 0.08
    spread_threshold: float = 0.2  # a cell spreads when its fire > this
    fire_increment: float = 0.12  # added to a newly-ignited cell

    # --- decay / fuel ---
    decay: float = 0.97
    extinguish_threshold: float = 0.02  # fire below this is zeroed
    fuel_depletion: float = 0.003

    # --- suppression (agent action) ---
    suppression_radius: int = 1  # 1 => 3x3 patch
    suppression_factor: float = 0.2  # fire in patch multiplied by this
    suppression_bonus: float = 2.0
    suppression_bonus_threshold: float = 0.1

    # --- termination ---
    termination_fire_threshold: float = 0.1

    # --- reward ---
    # "raw": -sum(fire) (preserves original behavior, NOT comparable across regions)
    # "normalized": reward divided by initial total fire (comparable across regions)
    reward_mode: str = "raw"

    # --- wind model ---
    # False => isotropic (original: (wind_x + wind_y) / 2, no direction)
    # True  => magnitude-weighted (experimental hook for directional wind)
    directional_wind: bool = False

    # --- scenario generation (fixes train/test leakage when enabled) ---
    # False => fixed ignition from the loaded tensor (original behavior)
    # True  => randomized ignition each reset, enabling disjoint train/eval scenarios
    randomize_ignition: bool = False
    n_ignition_points: int = 3
    ignition_intensity: float = 1.0


@dataclass
class MetricsConfig:
    """Single source of truth for evaluation metrics (fixes threshold drift)."""

    burned_threshold: float = 0.5  # a cell counts as "burned" when fire > this


@dataclass
class PPOConfig:
    """PPO + feature-extractor hyperparameters."""

    policy: str = "CnnPolicy"
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_envs: int = 1
    verbose: int = 0
    features_dim: int = 256
    cnn_pooling: bool = True  # adds spatial downsampling (shrinks 200MB checkpoints)


@dataclass
class RegionConfig:
    """A geographic region and its state tensor."""

    name: str = "saudi"
    dir: str = "saudi_eastern_province"  # subdir under data/
    grid_size: int = 32


@dataclass
class EvalConfig:
    n_episodes: int = 20
    deterministic: bool = True
    base_seed: int = 0


@dataclass
class MARLConfig:
    num_agents: int = 3
    agent_counts: list[int] = field(default_factory=lambda: [1, 3, 5])


@dataclass
class Config:
    """Top-level config composed from the sections above."""

    seed: int = 0
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    region: RegionConfig = field(default_factory=RegionConfig)
    # Used by multi-region experiments (e.g. the transfer matrix). Empty by default.
    regions: list[RegionConfig] = field(default_factory=list)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    marl: MARLConfig = field(default_factory=MARLConfig)
    experiment_name: str = "default"


def default_config() -> Config:
    """Return a fully-populated default config object."""
    return OmegaConf.to_object(OmegaConf.structured(Config))  # type: ignore[return-value]


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load a config: structured defaults <- YAML file <- dotlist overrides.

    Args:
        path: optional YAML file (e.g. ``configs/experiment/transfer.yaml``).
        overrides: optional dotlist, e.g. ``["ppo.total_timesteps=1000", "seed=1"]``.
    """
    base = OmegaConf.structured(Config)
    if path is not None:
        base = OmegaConf.merge(base, OmegaConf.load(str(path)))
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(base)  # type: ignore[return-value]


def to_dict(cfg: Config) -> dict[str, Any]:
    """Convert a config object to a plain dict (for hashing / metadata / logging)."""
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)  # type: ignore[return-value]
