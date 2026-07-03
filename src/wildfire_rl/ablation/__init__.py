"""Strategic ablation framework (Phase 15B.8).

Defines the ablation groups (single-factor cells over the full proposed hybrid system) and the
env/policy builders, so a runner can enumerate cells, evaluate them with the strategic metrics, and
emit provenance-bound CSVs + canonical rollout GIFs.
"""

from wildfire_rl.ablation.groups import GROUPS, build_env_factory, make_cell_policy

__all__ = ["GROUPS", "build_env_factory", "make_cell_policy"]
