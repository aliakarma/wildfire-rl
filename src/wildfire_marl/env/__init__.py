"""Environment layer (Phase 1+): Cell2Fire binding, Gymnasium single-agent env,
PettingZoo MARL env, and pluggable rewards. Cell2Fire physics is never modified —
all suppression/agent logic lives in the wrapper."""

from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.env.single_agent_env import FireSuppressionEnv

__all__ = ["FireSuppressionEnv", "MultiAgentFireEnv"]
