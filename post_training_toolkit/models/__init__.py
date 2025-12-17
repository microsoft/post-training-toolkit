"""Diagnostics engine for analyzing RLHF training logs.

Provides trainer-aware heuristics for detecting common failure modes:
- DPO: Loss at 0.693, margin collapse, win-rate instability
- PPO: Value head divergence, entropy collapse, advantage explosion
- SFT: Loss plateau, perplexity spikes
- ORPO: Odds ratio instability
- KTO: Desirable/undesirable imbalance
"""

from post_training_toolkit.models.engine import (
    run_diagnostics,
    load_jsonl,
    load_metrics,
    summarize_run,
    compute_derived_metrics,
)
from post_training_toolkit.models.heuristics import (
    run_heuristics,
    run_all_heuristics,
    Insight,
    TrainerType,
)

__all__ = [
    "run_diagnostics",
    "load_jsonl",
    "load_metrics",
    "summarize_run",
    "compute_derived_metrics",
    "run_heuristics",
    "run_all_heuristics",
    "Insight",
    "TrainerType",
]
