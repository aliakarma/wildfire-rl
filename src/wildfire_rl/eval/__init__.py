"""Evaluation: canonical metrics, baseline policies, evaluation loop, transfer matrix."""

from wildfire_rl.eval.baselines import NoOpPolicy, RandomPolicy
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.metrics import burned_cells, fire_intensity, summarize
from wildfire_rl.eval.transfer import transfer_matrix

__all__ = [
    "burned_cells",
    "fire_intensity",
    "summarize",
    "evaluate_policy",
    "transfer_matrix",
    "RandomPolicy",
    "NoOpPolicy",
]
