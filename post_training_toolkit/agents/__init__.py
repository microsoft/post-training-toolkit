"""Agent trace diagnostics and dataset conversion for TRL.

Turn agent run logs into health reports and training data.

Example:
    from post_training_toolkit.agents import AgentRunLog, analyze_runs, to_preference_pairs
    
    # Load traces
    runs = AgentRunLog.from_jsonl("agent_runs.jsonl")
    
    # Get diagnostics report
    report = analyze_runs(runs)
    print(report)
    
    # Convert to preference pairs for TRL DPO/KTO
    dataset = to_preference_pairs(
        runs,
        positive=lambda r: r.success and r.total_steps < 15,
        negative=lambda r: not r.success,
    )
    # dataset is an HF Dataset ready for DPOTrainer
"""

from post_training_toolkit.agents.traces import (
    AgentRunLog,
    Episode,
    Step,
    StepType,
)
from post_training_toolkit.agents.heuristics import (
    analyze_runs,
    AgentDiagnosticsReport,
    AgentInsight,
)
from post_training_toolkit.agents.converters import (
    to_preference_pairs,
    to_kto_dataset,
    to_sft_dataset,
    to_grpo_dataset,
    format_episode_as_conversation,
)
from post_training_toolkit.agents.loop import (
    AgentTrainingLoop,
    ComparisonResult,
)
from post_training_toolkit.agents.trajectory import (
    TrajectoryLogger,
    EpisodeHandle,
)

__all__ = [
    # Trace loading
    "AgentRunLog",
    "Episode",
    "Step",
    "StepType",
    # Diagnostics
    "analyze_runs",
    "AgentDiagnosticsReport",
    "AgentInsight",
    # Conversion
    "to_preference_pairs",
    "to_kto_dataset",
    "to_sft_dataset",
    "to_grpo_dataset",
    "format_episode_as_conversation",
    # Training loop
    "AgentTrainingLoop",
    "ComparisonResult",
    # Trajectory logging
    "TrajectoryLogger",
    "EpisodeHandle",
]
