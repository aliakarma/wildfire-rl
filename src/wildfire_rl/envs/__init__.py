"""Wildfire environments (single source of truth)."""

from wildfire_rl.envs.base import WildfireEnv, make_env_factory
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv

__all__ = ["WildfireEnv", "MultiAgentWildfireEnv", "make_env_factory"]
