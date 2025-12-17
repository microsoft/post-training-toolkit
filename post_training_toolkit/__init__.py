"""Post-Training Toolkit - Diagnostics & interpretability for RLHF training.

A lightweight, code-level toolkit for understanding and debugging post-training
runs (DPO, PPO, GRPO, SFT, ORPO, KTO, CPO) with Hugging Face TRL.

Features:
- **Auto-detecting callback**: Automatically detects trainer type and configures metrics
- **Trainer-specific heuristics**: Algorithm-aware diagnostics for each TRL trainer
- **Behavior snapshots**: Track model outputs on fixed prompts over training
- **Behavior diffing**: Compare snapshots to detect drift, refusal changes, etc.
- **Refusal detection**: Transparent regex/template-based refusal detection
- **Postmortem recording**: Automatic crash/interrupt diagnostics
- **Checkpoint comparison**: Score checkpoints and recommend the safest one
- **Resume validation**: Validate training run resumption from checkpoints
- **Experiment tracking**: Integration with WandB, MLflow, TensorBoard
- **Zero configuration**: Just add the callback and run diagnostics

Example:
    from post_training_toolkit import DiagnosticsCallback, run_diagnostics
    
    # Add callback to any TRL trainer - it auto-detects the trainer type
    trainer = DPOTrainer(
        model=model,
        callbacks=[DiagnosticsCallback(
            run_dir="my_run",
            enable_snapshots=True,  # Capture behavior snapshots
            snapshot_interval=100,
            experiment_tracker="wandb",  # Optional: stream to WandB
        )],
        ...
    )
    trainer.train()
    
    # Generate diagnostics report with trainer-aware analysis
    run_diagnostics(Path("my_run"), Path("reports"))

Supported Trainers:
- DPOTrainer: Loss at 0.693, margin collapse, win-rate instability
- PPOTrainer: Value head divergence, entropy collapse, advantage explosion  
- GRPOTrainer: Group reward collapse, advantage explosion, length drift
- SFTTrainer: Loss plateau, perplexity spikes
- ORPOTrainer: Odds ratio instability
- KTOTrainer: Desirable/undesirable imbalance
- CPOTrainer: Preference consistency issues
"""

from post_training_toolkit.integrations import DiagnosticsCallback, TrainerType
from post_training_toolkit.models import run_diagnostics, run_heuristics, Insight, load_metrics

# Artifact management
from post_training_toolkit.models.artifacts import (
    RunArtifactManager,
    RunMetadata,
    Snapshot,
    SnapshotEntry,
    Postmortem,
    # Provenance functions
    collect_full_provenance,
    get_git_info,
    get_hardware_info,
    get_package_versions,
    compute_config_hash,
    get_model_identity,
    get_tokenizer_identity,
    get_dataset_identity,
)

# Behavior snapshots and diffing
from post_training_toolkit.models.snapshots import (
    SnapshotManager,
    GenerationConfig,
    DEFAULT_EVAL_PROMPTS,
)
from post_training_toolkit.models.diffing import DiffManager, diff_snapshots, DiffThresholds

# Refusal detection
from post_training_toolkit.refusal import (
    RefusalDetector,
    RefusalResult,
    RefusalType,
    is_refusal,
    detect_refusal,
)

# Postmortem
from post_training_toolkit.models.postmortem import PostmortemRecorder

# Checkpoint comparison & resume validation
from post_training_toolkit.models.checkpoints import (
    CheckpointComparator,
    CheckpointRecommendation,
    recommend_checkpoint,
    ResumeValidator,
    ResumeValidationResult,
    validate_resume,
)

# Agent trace diagnostics
from post_training_toolkit.agents import (
    AgentRunLog,
    Episode,
    Step,
    StepType,
    analyze_runs,
    AgentDiagnosticsReport,
    AgentInsight,
    to_preference_pairs,
    to_kto_dataset,
    format_episode_as_conversation,
)

__version__ = "0.3.0"
__all__ = [
    # Core
    "DiagnosticsCallback",
    "TrainerType", 
    "run_diagnostics",
    "run_heuristics",
    "Insight",
    "load_metrics",
    # Artifacts
    "RunArtifactManager",
    "RunMetadata",
    "Snapshot",
    "SnapshotEntry",
    "Postmortem",
    # Provenance
    "collect_full_provenance",
    "get_git_info",
    "get_hardware_info",
    "get_package_versions",
    "compute_config_hash",
    "get_model_identity",
    "get_tokenizer_identity",
    "get_dataset_identity",
    # Snapshots & Diffing
    "SnapshotManager",
    "GenerationConfig",
    "DEFAULT_EVAL_PROMPTS",
    "DiffManager",
    "diff_snapshots",
    "DiffThresholds",
    # Refusal
    "RefusalDetector",
    "RefusalResult",
    "RefusalType",
    "is_refusal",
    "detect_refusal",
    # Postmortem
    "PostmortemRecorder",
    # Checkpoints & Resume
    "CheckpointComparator",
    "CheckpointRecommendation",
    "recommend_checkpoint",
    "ResumeValidator",
    "ResumeValidationResult",
    "validate_resume",
    # Agent Traces
    "AgentRunLog",
    "Episode",
    "Step",
    "StepType",
    "analyze_runs",
    "AgentDiagnosticsReport",
    "AgentInsight",
    "to_preference_pairs",
    "to_kto_dataset",
    "format_episode_as_conversation",
]
