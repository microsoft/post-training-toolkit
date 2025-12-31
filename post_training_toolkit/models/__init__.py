"""Diagnostics engine for analyzing RLHF training logs.

Provides trainer-aware heuristics for detecting common failure modes:
- DPO: Loss at 0.693, margin collapse, win-rate instability
- PPO: Value head divergence, entropy collapse, advantage explosion
- SFT: Loss plateau, perplexity spikes
- ORPO: Odds ratio instability
- KTO: Desirable/undesirable imbalance

Also provides distributed training utilities for multi-GPU profiling:
- Rank-aware logging
- Cross-rank metric aggregation
- Straggler detection
- Distributed memory tracking
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
from post_training_toolkit.models.distributed import (
    get_rank,
    get_local_rank,
    get_world_size,
    is_main_process,
    is_distributed,
    barrier,
    gather_dict,
    StragglerDetector,
    DistributedMemoryTracker,
)

__all__ = [
    # Engine
    "run_diagnostics",
    "load_jsonl",
    "load_metrics",
    "summarize_run",
    "compute_derived_metrics",
    # Heuristics
    "run_heuristics",
    "run_all_heuristics",
    "Insight",
    "TrainerType",
    # Distributed
    "get_rank",
    "get_local_rank",
    "get_world_size",
    "is_main_process",
    "is_distributed",
    "barrier",
    "gather_dict",
    "StragglerDetector",
    "DistributedMemoryTracker",
]
