"""Evaluation: canonical metrics, baseline policies, evaluation loop, transfer matrix,
and statistical significance testing."""

from wildfire_rl.eval.baselines import NoOpPolicy, RandomPolicy
from wildfire_rl.eval.evaluate import evaluate_policy
from wildfire_rl.eval.loading import load_ppo_model
from wildfire_rl.eval.metrics import burned_cells, fire_intensity, summarize
from wildfire_rl.eval.significance import (
    cohens_d,
    confidence_interval_95,
    format_ci,
    format_mean_std,
    format_significance,
    paired_ttest,
    welch_ttest,
    wilcoxon_test,
)
from wildfire_rl.eval.transfer import transfer_matrix

__all__ = [
    "burned_cells",
    "fire_intensity",
    "summarize",
    "evaluate_policy",
    "transfer_matrix",
    "RandomPolicy",
    "NoOpPolicy",
    "confidence_interval_95",
    "cohens_d",
    "welch_ttest",
    "paired_ttest",
    "wilcoxon_test",
    "format_ci",
    "format_mean_std",
    "format_significance",
    "load_ppo_model",
]
