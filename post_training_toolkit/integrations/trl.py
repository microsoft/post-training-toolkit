"""TRL integration callback for RLHF diagnostics logging.

Auto-detects trainer type (DPO, PPO, SFT, ORPO, KTO, CPO) and configures
metric collection accordingly. Zero configuration required.

The callback provides:
- Automatic trainer type detection
- Comprehensive provenance capture (git, config, hardware, packages)
- Structured artifact management (metrics, snapshots, postmortems)
- Behavior snapshot capture with default sentinel prompts
- Auto-diff behavior between snapshots
- Resume validation with configurable strictness
- Safe stopping on critical failures (opt-in)
- Crash/interrupt postmortem recording
- Distributed training support (only rank 0 writes)

North-star claim:
    "Add one callback to any TRL trainer and you get auditable runs, regression 
    visibility, safe resume, and optional fail-fast stopping—without writing glue code."

Usage:
    from post_training_toolkit import DiagnosticsCallback
    from trl import DPOTrainer  # or PPOTrainer, SFTTrainer, etc.

    # Just add the callback - everything happens automatically
    trainer = DPOTrainer(..., callbacks=[DiagnosticsCallback()])
    trainer.train()

    # By default you get:
    # - run_metadata_start.json: Immutable provenance at start
    # - run_metadata_final.json: Immutable provenance at end
    # - metrics.jsonl: Step-level training metrics
    # - snapshots/: Behavior snapshots at intervals (if enabled)
    # - diffs/: Behavior diffs between snapshots (auto when snapshots enabled)
    # - postmortem.json: Crash/interrupt diagnostics (if applicable)
    # - reports/: Auto-generated diagnostics report with insights and plots

    # Diagnostics run automatically. To disable:
    # trainer = DPOTrainer(..., callbacks=[DiagnosticsCallback(auto_diagnostics=False)])
    
    # Or run diagnostics manually on any run:
    # from post_training_toolkit import run_diagnostics
    # run_diagnostics(Path("diagnostic_run"), Path("reports"))
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import TrainingArguments

from post_training_toolkit.models.artifacts import (
    RunArtifactManager,
    is_main_process,
    compute_config_hash,
)
from post_training_toolkit.models.snapshots import (
    SnapshotManager,
    GenerationConfig,
    DEFAULT_EVAL_PROMPTS,
)
from post_training_toolkit.models.postmortem import PostmortemRecorder
from post_training_toolkit.refusal import RefusalDetector
from post_training_toolkit.models.engine import run_diagnostics
from post_training_toolkit.models.heuristics import run_heuristics, Insight, TrainerType
from post_training_toolkit.models.profiling import (
    StepTimer,
    SlowdownDetector,
    ThroughputTracker,
    GPUProfiler,
)
from post_training_toolkit.models.distributed import (
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process as distributed_is_main_process,
    gather_dict,
    StragglerDetector,
    DistributedMemoryTracker,
    get_distributed_info,
)


# Mapping from TRL trainer class names to our trainer types
TRAINER_CLASS_MAP = {
    "DPOTrainer": TrainerType.DPO,
    "PPOTrainer": TrainerType.PPO,
    "PPOv2Trainer": TrainerType.PPO,
    "SFTTrainer": TrainerType.SFT,
    "ORPOTrainer": TrainerType.ORPO,
    "KTOTrainer": TrainerType.KTO,
    "CPOTrainer": TrainerType.CPO,
    "GRPOTrainer": TrainerType.GRPO,
    # Also handle config classes
    "DPOConfig": TrainerType.DPO,
    "PPOConfig": TrainerType.PPO,
    "PPOv2Config": TrainerType.PPO,
    "SFTConfig": TrainerType.SFT,
    "ORPOConfig": TrainerType.ORPO,
    "KTOConfig": TrainerType.KTO,
    "CPOConfig": TrainerType.CPO,
    "GRPOConfig": TrainerType.GRPO,
}


class DiagnosticsCallback(TrainerCallback):
    """
    Auto-configuring callback that logs TRL training metrics for diagnostics.

    Features:
    - **Auto-detects trainer type** (DPO, PPO, SFT, ORPO, KTO, CPO)
    - **Trainer-specific metric collection** optimized for each algorithm
    - **Zero configuration required** - just add to callbacks list
    - **Compatible with all TRL trainers**

    The callback automatically:
    1. Detects which TRL trainer is being used
    2. Captures the relevant metrics for that trainer type
    3. Logs in a format optimized for post-hoc diagnostics
    4. Stores trainer metadata for algorithm-specific analysis
    """

    # Base metric mappings (common across trainers)
    BASE_METRIC_MAPPINGS = {
        # Reward metrics
        "reward_mean": ["ppo/mean_scores", "env/reward_mean", "rewards/mean", "reward", 
                        "objective/scores", "rewards/chosen", "train/reward"],
        "reward_std": ["ppo/std_scores", "env/reward_std", "rewards/std", "rewards/margins"],
        # KL divergence
        "kl": ["objective/kl", "ppo/mean_non_score_reward", "kl", "objective/kl_div", 
               "kl_div", "train/kl"],
        # Response quality
        "refusal_rate": ["refusal_rate", "safety/refusal_rate"],
        "output_length_mean": ["tokens/response_mean", "env/output_length", "response_length",
                               "train/response_length"],
        # Drift tracking
        "embedding_cosine_to_sft": ["policy/cosine_to_ref", "embedding_cosine", "cosine_to_sft"],
    }

    # DPO-specific metrics
    DPO_METRIC_MAPPINGS = {
        "dpo_loss": ["loss", "train_loss", "dpo_loss", "train/loss"],
        "win_rate": ["rewards/accuracies", "eval/win_rate", "win_rate", "train/rewards/accuracies"],
        "reward_margin": ["rewards/margins", "train/rewards/margins"],
        "logps_chosen": ["logps/chosen", "train/logps/chosen"],
        "logps_rejected": ["logps/rejected", "train/logps/rejected"],
        "logprobs": ["objective/logprobs", "logps/chosen"],
        # DPO-specific: beta-scaled implicit reward
        "rewards_chosen": ["rewards/chosen", "train/rewards/chosen"],
        "rewards_rejected": ["rewards/rejected", "train/rewards/rejected"],
    }

    # PPO-specific metrics
    PPO_METRIC_MAPPINGS = {
        "ppo_loss": ["ppo/loss/total", "ppo/loss", "loss/total", "train/ppo/loss/total"],
        "policy_loss": ["ppo/loss/policy", "ppo/policy_loss", "train/ppo/loss/policy"],
        "value_loss": ["ppo/loss/value", "ppo/value_loss", "train/ppo/loss/value"],
        "entropy": ["objective/entropy", "ppo/mean_entropy", "train/entropy"],
        "logprobs": ["objective/logprobs", "ppo/mean_logprobs"],
        # PPO-specific advantage and clipping
        "advantages_mean": ["ppo/mean_advantages", "advantages/mean"],
        "advantages_std": ["ppo/std_advantages", "advantages/std"],
        "clip_fraction": ["ppo/clip_fraction", "clip_fraction", "train/clip_fraction"],
        "approx_kl": ["ppo/approx_kl", "approx_kl"],
        # Value head metrics
        "value_mean": ["ppo/mean_values", "values/mean"],
        "value_std": ["ppo/std_values", "values/std"],
        "returns_mean": ["ppo/mean_returns", "returns/mean"],
        "returns_std": ["ppo/std_returns", "returns/std"],
    }

    # SFT-specific metrics
    SFT_METRIC_MAPPINGS = {
        "sft_loss": ["loss", "train_loss", "train/loss"],
        "perplexity": ["perplexity", "eval_perplexity", "train/perplexity"],
        "accuracy": ["accuracy", "eval_accuracy", "train/accuracy"],
    }

    # ORPO-specific metrics
    ORPO_METRIC_MAPPINGS = {
        "orpo_loss": ["loss", "train_loss", "train/loss"],
        "sft_loss": ["sft_loss", "train/sft_loss"],
        "odds_ratio_loss": ["odds_ratio_loss", "or_loss", "train/odds_ratio_loss"],
        "log_odds_ratio": ["log_odds_ratio", "train/log_odds_ratio"],
        "log_odds_chosen": ["log_odds_chosen", "train/log_odds_chosen"],
        "log_odds_rejected": ["log_odds_rejected", "train/log_odds_rejected"],
        "win_rate": ["rewards/accuracies", "eval/win_rate", "win_rate"],
    }

    # KTO-specific metrics
    KTO_METRIC_MAPPINGS = {
        "kto_loss": ["loss", "train_loss", "train/loss"],
        "kl": ["kl", "train/kl"],
        "logps_chosen": ["logps/chosen", "train/logps/chosen"],
        "logps_rejected": ["logps/rejected", "train/logps/rejected"],
        # KTO-specific
        "desirable_loss": ["desirable_loss", "train/desirable_loss"],
        "undesirable_loss": ["undesirable_loss", "train/undesirable_loss"],
    }

    # CPO-specific metrics
    CPO_METRIC_MAPPINGS = {
        "cpo_loss": ["loss", "train_loss", "train/loss"],
        "nll_loss": ["nll_loss", "train/nll_loss"],
        "win_rate": ["rewards/accuracies", "eval/win_rate", "win_rate"],
    }

    # GRPO-specific metrics (Group Relative Policy Optimization - DeepSeek)
    GRPO_METRIC_MAPPINGS = {
        "grpo_loss": ["loss", "train_loss", "train/loss", "grpo/loss"],
        "group_reward_mean": ["reward", "rewards/mean", "train/reward", "grpo/reward_mean"],
        "group_reward_std": ["rewards/std", "train/reward_std", "grpo/reward_std"],
        "group_advantage_mean": ["advantages/mean", "grpo/advantages_mean"],
        "group_advantage_std": ["advantages/std", "grpo/advantages_std"],
        "completion_length": ["completion_length", "train/completion_length", "grpo/completion_length"],
        "kl": ["kl", "train/kl", "objective/kl", "grpo/kl"],
        "entropy": ["entropy", "train/entropy", "objective/entropy", "grpo/entropy"],
        # GRPO-specific: group-level metrics
        "num_generations": ["num_generations", "grpo/num_generations"],
        "reward_per_func": ["reward_per_func", "grpo/reward_per_func"],
    }

    def __init__(
        self,
        run_dir: str | Path = "diagnostic_run",
        log_every_n_steps: int = 1,
        include_slices: bool = True,
        verbose: bool = False,
        # Snapshot options (default ON with sentinel prompts)
        enable_snapshots: bool = True,
        snapshot_interval: int = 100,
        snapshot_prompts: Optional[List[str]] = None,  # Uses DEFAULT_EVAL_PROMPTS if None
        snapshot_generation_config: Optional[Dict[str, Any]] = None,
        # Auto-diff options (default ON when snapshots enabled)
        enable_auto_diff: bool = True,
        # Postmortem options
        enable_postmortem: bool = True,
        # Resume validation options (default WARN)
        enable_resume_validation: bool = True,
        fail_on_resume_mismatch: bool = False,  # Default: warn only
        # Safe stopping options (default OFF for critical issues)
        stop_on_critical: bool = False,
        # Live warnings options (default ON)
        enable_live_warnings: bool = True,
        live_warning_interval: int = 10,  # Check heuristics every N steps
        # Provenance options (default ON)
        save_git_diff: bool = False,  # Only if dirty
        full_package_snapshot: bool = False,  # Full pip freeze
        # Experiment tracking options
        experiment_tracker: Optional[str] = None,
        experiment_name: Optional[str] = None,
        experiment_project: Optional[str] = None,
        experiment_tags: Optional[List[str]] = None,
        tracker_kwargs: Optional[Dict[str, Any]] = None,
        # Auto-diagnostics options (default ON - prints summary at end)
        auto_diagnostics: bool = True,
        # Distributed training options (auto-detected)
        distributed_metrics: bool = True,
        straggler_detection: bool = True,
        straggler_check_interval: int = 50,
        distributed_memory: bool = True,
        # YAML heuristics options
        custom_alerts: Optional[List[str]] = None,
        custom_heuristics_dir: Optional[str | Path] = None,
        disable_yaml_heuristics: bool = False,
        # Legacy compatibility
        log_path: Optional[str | Path] = None,
    ):
        """
        Initialize the diagnostics callback.

        By default, this callback provides:
        - Comprehensive provenance capture (git, config hash, packages, hardware)
        - Trainer-specific metric collection (auto-detected)
        - Behavior snapshots on default sentinel prompts
        - Auto-diff between consecutive snapshots
        - Resume validation with warnings
        - Crash/interrupt postmortem recording

        Args:
            run_dir: Directory for all run artifacts. Default: "diagnostic_run"
            log_every_n_steps: Log frequency (1 = every step). Default: 1
            include_slices: Whether to capture slice:* metrics if present. Default: True
            verbose: Print logged metrics to stdout. Default: False
            
            # Snapshot options (default ON)
            enable_snapshots: Enable behavior snapshot capture. Default: True
            snapshot_interval: Capture snapshots every N steps. Default: 100
            snapshot_prompts: Custom prompts for snapshots (uses 13 default sentinel prompts if None)
            snapshot_generation_config: Config for snapshot generation (deterministic by default)
            
            # Auto-diff options (default ON)
            enable_auto_diff: Auto-diff consecutive snapshots. Default: True
            
            # Postmortem options
            enable_postmortem: Enable crash/interrupt postmortem. Default: True
            
            # Resume validation options
            enable_resume_validation: Validate config consistency on resume. Default: True
            fail_on_resume_mismatch: Raise error on resume mismatch (vs warn). Default: False
            
            # Safe stopping options (default ON)
            stop_on_critical: Stop training on critical failures (NaN, high-severity issues). Default: True
            
            # Live warnings options (default ON)
            enable_live_warnings: Print warnings during training when issues detected. Default: True
            live_warning_interval: Run heuristics every N steps. Default: 10
            
            # Provenance options
            save_git_diff: Save uncommitted git changes to file. Default: False
            full_package_snapshot: Include all installed packages. Default: False
            
            # Experiment tracking
            experiment_tracker: Tracker type ("wandb", "mlflow", "tensorboard", or None)
            experiment_name: Name for the experiment run
            experiment_project: Project name (for WandB) or experiment name (for MLflow)
            experiment_tags: Tags for the experiment run
            tracker_kwargs: Additional kwargs passed to tracker initialization
            
            # Auto-diagnostics (prints summary to console at end)
            auto_diagnostics: Print diagnostics summary at training end. Default: True
            
            # Distributed training options (auto-detected, zero config needed)
            distributed_metrics: Aggregate metrics across ranks. Default: True (auto-enabled if distributed)
            straggler_detection: Detect slow ranks in distributed training. Default: True
            straggler_check_interval: Check for stragglers every N steps. Default: 50
            distributed_memory: Track memory across ranks. Default: True

            # YAML heuristics options (requires pyyaml: pip install pyyaml)
            custom_alerts: Inline alert strings for quick custom heuristics.
                Example: ["dpo: margin < 0.1 -> high: Margin collapsed"]
            custom_heuristics_dir: Directory containing custom YAML heuristic files
            disable_yaml_heuristics: Disable all YAML-based heuristics. Default: False

            log_path: DEPRECATED - use run_dir instead

        Note:
            Trainer type is auto-detected at training start. No need to specify manually.
            Only rank 0 writes artifacts in distributed training.
        """
        # Handle legacy log_path argument
        if log_path is not None:
            warnings.warn(
                "log_path is deprecated, use run_dir instead",
                DeprecationWarning,
                stacklevel=2,
            )
            # Legacy mode: use log_path's parent as run_dir
            self.run_dir = Path(log_path).parent
        else:
            self.run_dir = Path(run_dir)
        
        self.log_every_n_steps = log_every_n_steps
        self.include_slices = include_slices
        self.verbose = verbose
        
        # Snapshot config (default ON with sentinel prompts)
        self.enable_snapshots = enable_snapshots
        self.snapshot_interval = snapshot_interval
        self.snapshot_prompts = snapshot_prompts  # None = use DEFAULT_EVAL_PROMPTS
        self.snapshot_generation_config = snapshot_generation_config
        
        # Auto-diff config (default ON)
        self.enable_auto_diff = enable_auto_diff
        
        # Postmortem config
        self.enable_postmortem = enable_postmortem
        
        # Resume validation config
        self.enable_resume_validation = enable_resume_validation
        self.fail_on_resume_mismatch = fail_on_resume_mismatch
        
        # Safe stopping config (default ON)
        self.stop_on_critical = stop_on_critical
        self._critical_failure_detected = False
        self._stop_reason: Optional[str] = None
        self._last_good_step: int = 0
        
        # Live warnings config (default ON)
        self.enable_live_warnings = enable_live_warnings
        self.live_warning_interval = live_warning_interval
        self._metrics_history: List[Dict[str, Any]] = []  # Ring buffer for heuristics
        self._max_history_size = 200  # Keep last 200 steps
        self._warned_issues: Set[str] = set()  # Avoid spamming same warning
        
        # Provenance config
        self.save_git_diff = save_git_diff
        self.full_package_snapshot = full_package_snapshot
        
        # Experiment tracker config
        self._experiment_tracker_type = experiment_tracker
        self._experiment_name = experiment_name
        self._experiment_project = experiment_project
        self._experiment_tags = experiment_tags
        self._tracker_kwargs = tracker_kwargs or {}
        
        # Auto-diagnostics config (default ON - prints summary to console)
        self.auto_diagnostics = auto_diagnostics
        
        # State (set during training)
        self._initialized = False
        self._trainer_type: str = TrainerType.UNKNOWN
        self._metric_mappings: Dict[str, List[str]] = {}
        self._is_main = is_main_process()
        self._previous_snapshot_step: Optional[int] = None
        self._resume_validation_result: Optional[Dict[str, Any]] = None
        
        # Managers (initialized on train_begin)
        self._artifact_manager: Optional[RunArtifactManager] = None
        self._snapshot_manager: Optional[SnapshotManager] = None
        self._diff_manager: Optional[Any] = None
        self._postmortem_recorder: Optional[PostmortemRecorder] = None
        self._experiment_tracker: Optional[Any] = None
        self._trainer_ref = None  # Weak reference to trainer for snapshots
        
        # Profiling components (always enabled, minimal overhead)
        self._step_timer = StepTimer(window_size=50)
        self._slowdown_detector = SlowdownDetector(
            threshold=1.5,  # Warn if 50% slower
            severe_threshold=2.0,  # Severe if 2x slower
            min_steps_for_baseline=50,
            check_interval=20,  # Check every 20 steps
        )
        self._throughput_tracker = ThroughputTracker(window_size=100)
        self._gpu_profiler = GPUProfiler()
        
        # Distributed training components (auto-enabled if distributed detected)
        self._distributed_metrics = distributed_metrics
        self._straggler_detection = straggler_detection
        self._straggler_check_interval = straggler_check_interval
        self._distributed_memory = distributed_memory

        # YAML heuristics options
        self._custom_alerts = custom_alerts
        self._custom_heuristics_dir = Path(custom_heuristics_dir) if custom_heuristics_dir else None
        self._disable_yaml_heuristics = disable_yaml_heuristics

        # Initialize distributed components (will be set up properly on train_begin)
        self._straggler_detector: Optional[StragglerDetector] = None
        self._distributed_memory_tracker: Optional[DistributedMemoryTracker] = None
        self._is_distributed = False  # Set on train_begin
        self._world_size = 1
        self._rank = 0

    @property
    def log_path(self) -> Path:
        """Legacy compatibility: return metrics.jsonl path."""
        if self._artifact_manager:
            return self._artifact_manager.metrics_path
        return self.run_dir / "metrics.jsonl"

    def _detect_trainer_type(self, **kwargs) -> str:
        """Auto-detect trainer type from the trainer instance or args."""
        # Try to get trainer from kwargs
        trainer = kwargs.get("trainer")
        if trainer is not None:
            class_name = type(trainer).__name__
            if class_name in TRAINER_CLASS_MAP:
                return TRAINER_CLASS_MAP[class_name]
        
        # Try to detect from args class name
        args = kwargs.get("args")
        if args is not None:
            args_class = type(args).__name__
            if args_class in TRAINER_CLASS_MAP:
                return TRAINER_CLASS_MAP[args_class]
        
        # Try to detect from model config or other signals
        model = kwargs.get("model")
        if model is not None:
            # Check if model has value_head (PPO-specific)
            if hasattr(model, "v_head") or hasattr(model, "value_head"):
                return TrainerType.PPO
        
        return TrainerType.UNKNOWN
    
    def _determine_is_main_process(self, state: TrainerState, trainer: Optional[Any]) -> bool:
        """Use trainer/state/accelerate flags to decide if this is the main process."""
        # Prefer TrainerState flag if available
        if hasattr(state, "is_world_process_zero"):
            try:
                return bool(state.is_world_process_zero)
            except Exception:
                pass
        
        # Transformers Trainer exposes is_world_process_zero attribute or method
        if trainer is not None:
            if hasattr(trainer, "is_world_process_zero"):
                attr = trainer.is_world_process_zero
                try:
                    return bool(attr() if callable(attr) else attr)
                except Exception:
                    pass
            # Accelerate exposes accelerator.is_main_process
            accelerator = getattr(trainer, "accelerator", None)
            if accelerator is not None and hasattr(accelerator, "is_main_process"):
                try:
                    return bool(accelerator.is_main_process)
                except Exception:
                    pass
        
        return is_main_process()

    def _build_metric_mappings(self, trainer_type: str) -> Dict[str, List[str]]:
        """Build the metric mapping dict based on detected trainer type."""
        mappings = dict(self.BASE_METRIC_MAPPINGS)
        
        if trainer_type == TrainerType.DPO:
            mappings.update(self.DPO_METRIC_MAPPINGS)
        elif trainer_type == TrainerType.PPO:
            mappings.update(self.PPO_METRIC_MAPPINGS)
        elif trainer_type == TrainerType.SFT:
            mappings.update(self.SFT_METRIC_MAPPINGS)
        elif trainer_type == TrainerType.ORPO:
            mappings.update(self.ORPO_METRIC_MAPPINGS)
        elif trainer_type == TrainerType.KTO:
            mappings.update(self.KTO_METRIC_MAPPINGS)
        elif trainer_type == TrainerType.CPO:
            mappings.update(self.CPO_METRIC_MAPPINGS)
        elif trainer_type == TrainerType.GRPO:
            mappings.update(self.GRPO_METRIC_MAPPINGS)
        else:
            # Unknown trainer - include all mappings for best coverage
            mappings.update(self.DPO_METRIC_MAPPINGS)
            mappings.update(self.PPO_METRIC_MAPPINGS)
            mappings.update(self.SFT_METRIC_MAPPINGS)
            mappings.update(self.GRPO_METRIC_MAPPINGS)
        
        return mappings

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize artifacts, auto-detect trainer type, and run pre-flight checks.
        
        This method:
        1. Auto-detects the TRL trainer type
        2. Collects comprehensive provenance (git, config, hardware, packages)
        3. Creates run directory and writes immutable start metadata
        4. Initializes snapshot manager with sentinel prompts
        5. Runs resume validation if this is a resumed run
        6. Sets up postmortem recording
        """
        # Auto-detect trainer type
        self._trainer_type = self._detect_trainer_type(args=args, **kwargs)
        is_main = self._determine_is_main_process(state, kwargs.get("trainer"))
        
        # Build metric mappings for this trainer type
        self._metric_mappings = self._build_metric_mappings(self._trainer_type)
        
        # Store trainer reference for snapshots
        self._trainer_ref = kwargs.get("trainer")
        
        # Extract model/tokenizer/dataset for provenance
        model = None
        ref_model = None
        tokenizer = None
        dataset = None
        model_name = None
        ref_model_name = None
        
        if self._trainer_ref is not None:
            if hasattr(self._trainer_ref, "model"):
                model = self._trainer_ref.model
                if hasattr(model, "name_or_path"):
                    model_name = model.name_or_path
                elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
                    model_name = model.config._name_or_path
            if hasattr(self._trainer_ref, "ref_model"):
                ref_model = self._trainer_ref.ref_model
                if ref_model and hasattr(ref_model, "config"):
                    ref_model_name = getattr(ref_model.config, "_name_or_path", None)
            if hasattr(self._trainer_ref, "tokenizer"):
                tokenizer = self._trainer_ref.tokenizer
            if hasattr(self._trainer_ref, "train_dataset"):
                dataset = self._trainer_ref.train_dataset
        
        # Initialize artifact manager
        self._artifact_manager = RunArtifactManager(
            self.run_dir,
            is_main_process_override=is_main,
        )
        self._is_main = is_main
        
        # Build config dict with full training args
        config = {
            "log_every_n_steps": self.log_every_n_steps,
            "include_slices": self.include_slices,
            "enable_snapshots": self.enable_snapshots,
            "snapshot_interval": self.snapshot_interval,
            "enable_auto_diff": self.enable_auto_diff,
            "enable_resume_validation": self.enable_resume_validation,
            "fail_on_resume_mismatch": self.fail_on_resume_mismatch,
            "stop_on_critical": self.stop_on_critical,
        }
        if hasattr(args, "to_dict"):
            config["training_args"] = {
                k: v for k, v in args.to_dict().items()
                if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))
            }
        
        # Run resume validation if resuming from checkpoint
        is_resuming = state.global_step > 0
        if is_resuming and self.enable_resume_validation and is_main:
            self._run_resume_validation(state, config)
        
        # Initialize run with comprehensive provenance
        self._artifact_manager.initialize(
            trainer_type=self._trainer_type,
            model_name=model_name,
            ref_model_name=ref_model_name,
            config=config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            save_git_diff=self.save_git_diff,
            full_package_snapshot=self.full_package_snapshot,
        )
        
        # Initialize snapshot manager with default sentinel prompts
        if self.enable_snapshots:
            gen_config = None
            if self.snapshot_generation_config:
                gen_config = GenerationConfig(**self.snapshot_generation_config)
            else:
                # Deterministic generation config by default for reproducibility
                gen_config = GenerationConfig(
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    seed=42,  # Deterministic
                )
            
            self._snapshot_manager = SnapshotManager(
                artifact_manager=self._artifact_manager,
                prompts=self.snapshot_prompts,  # Uses DEFAULT_EVAL_PROMPTS if None
                snapshot_interval=self.snapshot_interval,
                generation_config=gen_config,
            )
            
            # Initialize diff manager for auto-diff
            if self.enable_auto_diff:
                from post_training_toolkit.models.diffing import DiffManager
                self._diff_manager = DiffManager(self._artifact_manager)
        
        # Initialize postmortem recorder
        if self.enable_postmortem:
            self._postmortem_recorder = PostmortemRecorder(self._artifact_manager)
            self._postmortem_recorder.install()
        
        # Initialize experiment tracker if configured
        if self._experiment_tracker_type and self._is_main:
            try:
                from post_training_toolkit.integrations.trackers import get_tracker
                
                tracker_kwargs = dict(self._tracker_kwargs)
                if self._experiment_tracker_type == "wandb":
                    tracker_kwargs.setdefault("project", self._experiment_project or "post-training-toolkit")
                    tracker_kwargs.setdefault("name", self._experiment_name)
                    if self._experiment_tags:
                        tracker_kwargs.setdefault("tags", self._experiment_tags)
                    tracker_kwargs.setdefault("config", config)
                elif self._experiment_tracker_type == "mlflow":
                    tracker_kwargs.setdefault("experiment_name", self._experiment_project or "post-training-toolkit")
                    tracker_kwargs.setdefault("run_name", self._experiment_name)
                    if self._experiment_tags:
                        tracker_kwargs.setdefault("tags", {f"tag_{i}": t for i, t in enumerate(self._experiment_tags)})
                elif self._experiment_tracker_type in ("tensorboard", "tb"):
                    tracker_kwargs.setdefault("log_dir", str(self.run_dir / "tensorboard"))
                    tracker_kwargs.setdefault("comment", self._experiment_name)
                
                self._experiment_tracker = get_tracker(self._experiment_tracker_type, **tracker_kwargs)
                
                # Log config to tracker
                self._experiment_tracker.log_config({
                    "trainer_type": self._trainer_type,
                    "model_name": model_name,
                    "ref_model_name": ref_model_name,
                    **config,
                })
                
                if self.verbose:
                    print(f"[DiagnosticsCallback] Experiment tracker: {self._experiment_tracker_type}")
            except ImportError as e:
                if self.verbose:
                    print(f"[DiagnosticsCallback] Tracker {self._experiment_tracker_type} not available: {e}")
                self._experiment_tracker = None
        
        # Initialize distributed training components (auto-detect)
        self._is_distributed = is_distributed()
        self._world_size = get_world_size()
        self._rank = get_rank()
        
        if self._is_distributed:
            if self._straggler_detection:
                self._straggler_detector = StragglerDetector(
                    window_size=50,
                    straggler_threshold=1.20,  # 20% slower = straggler
                    consistent_checks=3,
                )
            if self._distributed_memory:
                self._distributed_memory_tracker = DistributedMemoryTracker(history_size=100)
            
            if self.verbose and self._is_main:
                dist_info = get_distributed_info()
                print(f"[DiagnosticsCallback] Distributed training detected!")
                print(f"[DiagnosticsCallback]   World size: {self._world_size}")
                print(f"[DiagnosticsCallback]   Backend: {dist_info.backend}")
                if self._straggler_detection:
                    print(f"[DiagnosticsCallback]   Straggler detection: ON (check every {self._straggler_check_interval} steps)")
                if self._distributed_memory:
                    print(f"[DiagnosticsCallback]   Distributed memory tracking: ON")
        
        self._initialized = True
        
        if self.verbose and self._is_main:
            print(f"[DiagnosticsCallback] Detected trainer: {self._trainer_type.upper()}")
            print(f"[DiagnosticsCallback] Run directory: {self.run_dir}")
            if self.enable_snapshots:
                print(f"[DiagnosticsCallback] Snapshots enabled every {self.snapshot_interval} steps")
            if self.enable_auto_diff:
                print(f"[DiagnosticsCallback] Auto-diff enabled")
            if self.enable_postmortem:
                print(f"[DiagnosticsCallback] Postmortem recording enabled")
            if self.stop_on_critical:
                print(f"[DiagnosticsCallback] Safe stopping enabled (will stop on NaN/Inf)")

    def _run_resume_validation(self, state: TrainerState, new_config: Dict[str, Any]) -> None:
        """Run resume validation and emit warnings/errors.
        
        Validates that:
        - Config hash matches original (warns on mismatch)
        - Step continuity is maintained
        - Critical parameters haven't changed
        
        Results are recorded in artifacts and optionally fail training.
        """
        validation_result: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resumed_from_step": state.global_step,
            "warnings": [],
            "errors": [],
            "is_valid": True,
        }
        
        # Load original start metadata if it exists
        start_metadata_path = self._artifact_manager.metadata_start_path
        if start_metadata_path.exists():
            try:
                with open(start_metadata_path, "r") as f:
                    original = json.load(f)
                
                original_config_hash = original.get("config_hash")
                new_config_hash = compute_config_hash(new_config)
                
                if original_config_hash and new_config_hash != original_config_hash:
                    validation_result["warnings"].append(
                        f"Config hash mismatch: original={original_config_hash}, "
                        f"current={new_config_hash}. Training config may have changed."
                    )
                
                # Check critical parameters
                original_training_args = original.get("config", {}).get("training_args", {})
                new_training_args = new_config.get("training_args", {})
                
                critical_params = [
                    "learning_rate", "per_device_train_batch_size",
                    "gradient_accumulation_steps", "max_steps", "num_train_epochs",
                ]
                
                for param in critical_params:
                    orig_val = original_training_args.get(param)
                    new_val = new_training_args.get(param)
                    if orig_val is not None and new_val is not None and orig_val != new_val:
                        validation_result["warnings"].append(
                            f"Critical param '{param}' changed: {orig_val} → {new_val}"
                        )
                
                # Check model identity
                if original.get("model_name") and original.get("model_name") != new_config.get("model_name"):
                    validation_result["errors"].append(
                        f"Model mismatch: original={original.get('model_name')}"
                    )
                    validation_result["is_valid"] = False
                    
            except (json.JSONDecodeError, IOError) as e:
                validation_result["warnings"].append(
                    f"Could not load original metadata for validation: {e}"
                )
        else:
            validation_result["warnings"].append(
                "No run_metadata_start.json found - cannot validate resume consistency"
            )
        
        # Store result
        self._resume_validation_result = validation_result
        
        # Emit warnings/errors
        if self._is_main:
            for warning in validation_result["warnings"]:
                warnings.warn(f"[Resume Validation] {warning}", UserWarning, stacklevel=3)
                if self.verbose:
                    print(f"[DiagnosticsCallback] ⚠️ Resume warning: {warning}")
            
            for error in validation_result["errors"]:
                if self.verbose:
                    print(f"[DiagnosticsCallback] ❌ Resume error: {error}")
        
        # Save validation result as artifact
        if self._artifact_manager and self._is_main:
            validation_path = self._artifact_manager.run_dir / "resume_validation.json"
            with open(validation_path, "w") as f:
                json.dump(validation_result, f, indent=2)
        
        # Fail if configured and there are errors
        if self.fail_on_resume_mismatch and not validation_result["is_valid"]:
            raise RuntimeError(
                f"Resume validation failed: {validation_result['errors']}. "
                f"Set fail_on_resume_mismatch=False to continue with warnings."
            )

    def _check_critical_failure(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Check metrics for critical failures (NaN, Inf, divergence).
        
        Returns failure reason string if critical failure detected, None otherwise.
        """
        # Check for NaN/Inf in loss metrics
        loss_keys = ["loss", "dpo_loss", "ppo_loss", "sft_loss", "train_loss"]
        for key in loss_keys:
            if key in metrics:
                val = metrics[key]
                if math.isnan(val):
                    return f"NaN detected in {key}"
                if math.isinf(val):
                    return f"Inf detected in {key}"
        
        # Check for extreme loss values (likely divergence)
        for key in loss_keys:
            if key in metrics:
                val = metrics[key]
                if val > 100:  # Likely diverging
                    return f"Loss {key}={val} exceeds safe threshold (100)"
        
        # Check for NaN in any metric
        for key, val in metrics.items():
            if isinstance(val, float):
                if math.isnan(val):
                    return f"NaN detected in {key}"
                if math.isinf(val):
                    return f"Inf detected in {key}"
        
        return None

    def _handle_critical_failure(
        self,
        reason: str,
        step: int,
        control: TrainerControl,
    ) -> None:
        """Handle a critical failure - record and optionally stop training.
        
        Args:
            reason: Description of the failure
            step: Step at which failure occurred
            control: TrainerControl to set should_training_stop
        """
        self._critical_failure_detected = True
        self._stop_reason = reason
        
        if self._is_main:
            print(f"\n[DiagnosticsCallback] ❌ CRITICAL FAILURE at step {step}: {reason}")
            print(f"[DiagnosticsCallback] Last good step: {self._last_good_step}")
            
            if self.stop_on_critical:
                print(f"[DiagnosticsCallback] Stopping training (stop_on_critical=True)")
                print(f"[DiagnosticsCallback] Recommend resuming from checkpoint at step {self._last_good_step}")
        
        # Record in postmortem
        if self._postmortem_recorder:
            self._postmortem_recorder.record_event(
                "critical_failure",
                {
                    "reason": reason,
                    "step": step,
                    "last_good_step": self._last_good_step,
                }
            )
        
        # Save failure info to artifact
        if self._artifact_manager and self._is_main:
            failure_info = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "step": step,
                "last_good_step": self._last_good_step,
                "recommendation": f"Resume from checkpoint at step {self._last_good_step}",
            }
            failure_path = self._artifact_manager.run_dir / "critical_failure.json"
            with open(failure_path, "w") as f:
                json.dump(failure_info, f, indent=2)
        
        # Stop training if configured
        if self.stop_on_critical:
            control.should_training_stop = True

    def _run_live_heuristics(self, step: int, metrics: Dict[str, Any]) -> List[Insight]:
        """Run lightweight heuristics on recent metrics and return any insights.

        This runs trainer-appropriate heuristics on accumulated metrics history,
        optimized for low overhead during training. Includes both Python-based
        and YAML-based heuristics (if pyyaml is installed).
        """
        import pandas as pd

        # Add current metrics to history
        self._metrics_history.append({"step": step, **metrics})

        # Trim history to max size (ring buffer)
        if len(self._metrics_history) > self._max_history_size:
            self._metrics_history = self._metrics_history[-self._max_history_size:]

        # Need at least 10 steps for meaningful analysis
        if len(self._metrics_history) < 10:
            return []

        # Convert to DataFrame for heuristics
        df = pd.DataFrame(self._metrics_history)

        # Build custom_yaml_dirs list
        custom_yaml_dirs = None
        if self._custom_heuristics_dir:
            custom_yaml_dirs = [str(self._custom_heuristics_dir)]

        # Run heuristics for current trainer type (includes YAML heuristics)
        insights = run_heuristics(
            df,
            self._trainer_type,
            custom_alerts=self._custom_alerts,
            custom_yaml_dirs=custom_yaml_dirs,
            disable_yaml_heuristics=self._disable_yaml_heuristics,
        )

        return insights

    def _handle_live_insights(
        self,
        insights: List[Insight],
        step: int,
        control: TrainerControl,
    ) -> None:
        """Handle insights from live heuristics - warn and optionally stop."""
        if not insights:
            return
        
        for insight in insights:
            # Create unique key for this issue to avoid spam
            issue_key = f"{insight.type}_{insight.severity}"
            
            # For high severity, always warn (but rate-limit to every 50 steps)
            # For medium/low, only warn once per issue type
            should_warn = False
            if insight.severity == "high":
                step_key = f"{issue_key}_{step // 50}"
                if step_key not in self._warned_issues:
                    should_warn = True
                    self._warned_issues.add(step_key)
            else:
                if issue_key not in self._warned_issues:
                    should_warn = True
                    self._warned_issues.add(issue_key)
            
            if should_warn and self._is_main:
                severity_icon = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}.get(insight.severity, "•")
                print(f"\n[DiagnosticsCallback] {severity_icon} {insight.severity.upper()} at step {step}: {insight.message}")
                
                if insight.data:
                    # Print key diagnostic data (up to 3 items)
                    data_items = []
                    for k, v in list(insight.data.items())[:3]:
                        if isinstance(v, float):
                            data_items.append(f"{k}={v:.4f}")
                        else:
                            data_items.append(f"{k}={v}")
                    print(f"[DiagnosticsCallback]    Data: {', '.join(data_items)}")
                
                if insight.reference:
                    print(f"[DiagnosticsCallback]    Ref: {insight.reference}")
            
            # Stop on high severity if configured
            if insight.severity == "high" and self.stop_on_critical:
                self._critical_failure_detected = True
                self._stop_reason = f"{insight.type}: {insight.message}"
                
                if self._is_main:
                    print(f"\n[DiagnosticsCallback] 🛑 STOPPING TRAINING due to high-severity issue")
                    print(f"[DiagnosticsCallback]    Reason: {insight.message}")
                    print(f"[DiagnosticsCallback]    Last good step: {self._last_good_step}")
                
                control.should_training_stop = True
                break  # Don't process more insights if stopping

    def _find_metric(self, logs: Dict[str, Any], target_key: str) -> Optional[float]:
        """Look up a metric by trying multiple possible TRL names."""
        candidates = self._metric_mappings.get(target_key, [target_key])
        for name in candidates:
            if name in logs:
                val = logs[name]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    return float(val)
        return None

    def _extract_slices(self, logs: Dict[str, Any]) -> Dict[str, float]:
        """Extract slice:* metrics from logs."""
        slices = {}
        for key, val in logs.items():
            if key.startswith("slice:") or key.startswith("eval/slice_"):
                # Normalize key to slice:name format
                name = key.replace("eval/slice_", "slice:")
                if isinstance(val, (int, float)) and not math.isnan(val):
                    slices[name] = float(val)
        return slices

    def _extract_all_raw_metrics(self, logs: Dict[str, Any]) -> Dict[str, float]:
        """Extract all numeric metrics from logs (for discovery)."""
        raw = {}
        for key, val in logs.items():
            if isinstance(val, (int, float)) and not math.isnan(val):
                raw[key] = float(val)
        return raw

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Mark the start of a training step for profiling."""
        if not self._initialized:
            return
        
        # Start step timer for profiling
        self._step_timer.start_step(state.global_step)
        self._throughput_tracker.start_step()
        
        # Start straggler detection timing (distributed)
        if self._straggler_detector:
            self._straggler_detector.start_step()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Capture and write metrics on each log event.
        
        Also checks for critical failures (NaN, Inf) and handles safe stopping.
        In distributed mode, aggregates metrics across all ranks.
        """
        if logs is None or not self._initialized:
            return

        step = state.global_step
        if step % self.log_every_n_steps != 0:
            return

        # Build metrics dict
        metrics: Dict[str, Any] = {}

        # Core metrics (mapped to standard names)
        for target_key in self._metric_mappings:
            val = self._find_metric(logs, target_key)
            if val is not None:
                metrics[target_key] = round(val, 6)

        # Slice metrics
        if self.include_slices:
            slices = self._extract_slices(logs)
            metrics.update(slices)

        # Only log if we have meaningful data
        if not metrics:
            return
        
        # Aggregate metrics across ranks in distributed mode
        if self._is_distributed and self._distributed_metrics:
            # Gather metrics from all ranks (this is a collective operation)
            aggregated = gather_dict(metrics)
            
            # Replace local metrics with aggregated version for logging
            # Keep original local metrics for critical failure check (each rank checks its own)
            metrics_to_log = aggregated
            
            # Add distributed metadata
            metrics_to_log["world_size"] = self._world_size
        else:
            metrics_to_log = metrics
        
        # Check for critical failures (NaN, Inf, divergence) on LOCAL metrics
        # Each rank should check its own metrics for NaN/Inf
        critical_failure = self._check_critical_failure(metrics)
        if critical_failure:
            self._handle_critical_failure(critical_failure, step, control)
        else:
            # Update last good step (metrics were healthy)
            self._last_good_step = step

        # Log to artifact manager (only rank 0 writes)
        if self._artifact_manager:
            self._artifact_manager.log_metrics(step, metrics_to_log)
        
        # Log to experiment tracker
        if self._experiment_tracker and self._is_main:
            try:
                self._experiment_tracker.log_metrics(metrics_to_log, step=step)
            except Exception as e:
                if self.verbose:
                    print(f"[DiagnosticsCallback] Tracker log failed: {e}")
        
        # Record in postmortem buffer
        if self._postmortem_recorder:
            self._postmortem_recorder.record_step(step)
            self._postmortem_recorder.record_metrics(step, metrics_to_log)
            
            # Check for NaN/divergence
            self._postmortem_recorder.check_for_nan(metrics)
            self._postmortem_recorder.check_for_divergence(metrics)
        
        # Run live heuristics and warn/stop if issues detected
        if self.enable_live_warnings and step % self.live_warning_interval == 0:
            try:
                insights = self._run_live_heuristics(step, metrics)
                self._handle_live_insights(insights, step, control)
            except Exception as e:
                if self.verbose:
                    print(f"[DiagnosticsCallback] Live heuristics failed: {e}")
        else:
            # Still track metrics history even if not running heuristics this step
            self._metrics_history.append({"step": step, **metrics})
            if len(self._metrics_history) > self._max_history_size:
                self._metrics_history = self._metrics_history[-self._max_history_size:]

        if self.verbose and self._is_main:
            print(f"[DiagnosticsCallback] Step {step} ({self._trainer_type}): {list(metrics.keys())}")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Handle end of training step - profiling, snapshots, auto-diff, and distributed analysis."""
        if not self._initialized:
            return
        
        step = state.global_step
        
        # Complete step timing and check for slowdowns
        gpu_memory_mb = self._gpu_profiler.get_current_memory_mb()
        step_timing = self._step_timer.end_step(memory_mb=gpu_memory_mb)
        
        # Record GPU state
        self._gpu_profiler.record_step(step)
        
        # Track throughput (try to get batch info from trainer)
        batch_size = getattr(args, "per_device_train_batch_size", None)
        seq_length = getattr(args, "max_seq_length", None) or getattr(args, "max_length", None)
        self._throughput_tracker.end_step(
            batch_size=batch_size,
            seq_length=seq_length,
        )
        
        # === DISTRIBUTED: Straggler detection ===
        if self._straggler_detector and step_timing:
            self._straggler_detector.end_step(step, step_timing.duration_sec)
            
            # Check for stragglers periodically
            if step % self._straggler_check_interval == 0 and step > 0:
                try:
                    straggler_report = self._straggler_detector.analyze()
                    if straggler_report and straggler_report.has_straggler:
                        if self._is_main:
                            print(f"\n[PTT] ⚠️ STRAGGLER DETECTED at step {step}")
                            print(f"[PTT]    Slowest rank: {straggler_report.slowest_rank} ({straggler_report.slowdown_factor:.2f}x slower)")
                            print(f"[PTT]    Consistent: {'Yes' if straggler_report.is_consistent else 'No'}")
                            print(f"[PTT]    Likely cause: {straggler_report.likely_cause}")
                            print(f"[PTT]    Suggestion: {straggler_report.suggestion}")
                        
                        # Log straggler info to metrics
                        if self._artifact_manager:
                            self._artifact_manager.log_metrics(step, {
                                "straggler_rank": straggler_report.slowest_rank,
                                "straggler_slowdown_factor": straggler_report.slowdown_factor,
                                "distributed_efficiency": 1.0 / straggler_report.slowdown_factor,
                            })
                except Exception as e:
                    if self.verbose and self._is_main:
                        print(f"[DiagnosticsCallback] Straggler analysis failed: {e}")
        
        # === DISTRIBUTED: Memory tracking ===
        if self._distributed_memory_tracker and step % 100 == 0:
            try:
                self._distributed_memory_tracker.record(step)
                
                if self._distributed_memory_tracker.has_memory_issue():
                    report = self._distributed_memory_tracker.report()
                    if self._is_main:
                        print(f"\n[PTT] ⚠️ DISTRIBUTED MEMORY ISSUE at step {step}")
                        print(f"[PTT]    Highest memory rank: {report.highest_growth_rank}")
                        print(f"[PTT]    Memory imbalance: {report.current_snapshot.imbalance_ratio:.1%}")
                        if report.predicted_oom_rank is not None:
                            print(f"[PTT]    ⚠️ Predicted OOM: Rank {report.predicted_oom_rank}")
                    
                    # Log to metrics
                    if self._artifact_manager:
                        self._artifact_manager.log_metrics(step, {
                            "memory_imbalance_ratio": report.current_snapshot.imbalance_ratio,
                            "memory_max_rank": report.current_snapshot.max_rank,
                            "memory_max_mb": report.current_snapshot.max_mb,
                        })
            except Exception as e:
                if self.verbose and self._is_main:
                    print(f"[DiagnosticsCallback] Distributed memory tracking failed: {e}")
        
        # Check for slowdowns (single-process)
        slowdown_event = self._slowdown_detector.check(self._step_timer)
        if slowdown_event and self._is_main:
            print(f"\n[PTT] ⚠️ SLOWDOWN DETECTED at step {step}")
            print(f"[PTT]    Step time: {slowdown_event.baseline_duration:.2f}s → {slowdown_event.current_duration:.2f}s ({slowdown_event.slowdown_factor:.1f}x)")
            print(f"[PTT]    Cause: {slowdown_event.likely_cause}")
            print(f"[PTT]    Suggestion: {slowdown_event.suggestion}")
            if slowdown_event.memory_growth_mb:
                print(f"[PTT]    Memory growth: {slowdown_event.memory_growth_mb:+,.0f} MB")
            
            # Log slowdown to metrics
            if self._artifact_manager:
                self._artifact_manager.log_metrics(step, {
                    "slowdown_factor": slowdown_event.slowdown_factor,
                    "step_time_sec": slowdown_event.current_duration,
                    "baseline_step_time_sec": slowdown_event.baseline_duration,
                })
        
        # Snapshot capture
        if self.enable_snapshots and self._snapshot_manager and self._snapshot_manager.should_snapshot(step):
            # Try to get model and tokenizer from trainer
            trainer = self._trainer_ref
            if trainer is not None and hasattr(trainer, "model") and hasattr(trainer, "tokenizer"):
                try:
                    self._snapshot_manager.capture(
                        step=step,
                        model=trainer.model,
                        tokenizer=trainer.tokenizer,
                    )
                    if self.verbose and self._is_main:
                        print(f"[DiagnosticsCallback] Captured snapshot at step {step}")
                    
                    # Auto-diff against previous snapshot
                    if self.enable_auto_diff and self._diff_manager and self._previous_snapshot_step is not None:
                        try:
                            result = self._diff_manager.diff_steps(
                                self._previous_snapshot_step,
                                step,
                                save=True,
                            )
                            if result and self._is_main:
                                diff, summary = result
                                if self.verbose:
                                    print(f"[DiagnosticsCallback] Auto-diff: {self._previous_snapshot_step} → {step}")
                                    print(f"[DiagnosticsCallback]   Drift severity: {summary.drift_severity}")
                                    if summary.refusal_gained > 0:
                                        print(f"[DiagnosticsCallback]   ⚠️ {summary.refusal_gained} prompts started refusing")
                                    if summary.significant_changes > 0:
                                        print(f"[DiagnosticsCallback]   {summary.significant_changes} significant changes")
                                
                                # Log diff summary to metrics
                                if self._artifact_manager:
                                    self._artifact_manager.log_metrics(step, {
                                        "diff_severity": {"minimal": 0, "moderate": 1, "significant": 2}.get(summary.drift_severity, 0),
                                        "diff_significant_changes": summary.significant_changes,
                                        "diff_refusal_delta": summary.refusal_rate_after - summary.refusal_rate_before,
                                    })
                        except Exception as e:
                            if self.verbose and self._is_main:
                                print(f"[DiagnosticsCallback] Auto-diff failed: {e}")
                    
                    # Update previous snapshot step
                    self._previous_snapshot_step = step
                    
                except Exception as e:
                    if self.verbose and self._is_main:
                        print(f"[DiagnosticsCallback] Snapshot capture failed: {e}")
                    if self._postmortem_recorder:
                        self._postmortem_recorder.record_event(
                            "snapshot_failed", {"step": step, "error": str(e)}
                        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Finalize artifacts at training end."""
        status = "completed"
        
        # Check if we stopped due to critical failure
        if self._critical_failure_detected:
            status = "stopped_critical_failure"
            if self._is_main:
                print(f"[DiagnosticsCallback] Run ended due to critical failure: {self._stop_reason}")
                print(f"[DiagnosticsCallback] Recommend resuming from step {self._last_good_step}")
        
        if self.verbose and self._is_main:
            print(f"[DiagnosticsCallback] Training complete ({self._trainer_type}). Artifacts in {self.run_dir}")
        
        # Compute final auto-diffs if enabled (fill any gaps)
        if self.enable_auto_diff and self._diff_manager and self._is_main:
            try:
                # Compute all consecutive diffs (will skip existing ones)
                self._diff_manager.compute_all_diffs(save=True)
                if self.verbose:
                    print(f"[DiagnosticsCallback] Computed all behavior diffs")
            except Exception as e:
                if self.verbose:
                    print(f"[DiagnosticsCallback] Final diff computation failed: {e}")
        
        # Finalize artifact manager (writes immutable final metadata)
        if self._artifact_manager:
            self._artifact_manager.finalize(
                status=status,
                total_steps=state.global_step,
            )
        
        # Mark postmortem as finalized (prevents writing on normal exit)
        if self._postmortem_recorder:
            self._postmortem_recorder.finalize()
        
        # Finalize experiment tracker
        if self._experiment_tracker and self._is_main:
            try:
                # Log final summary
                summary = {
                    "trainer_type": self._trainer_type,
                    "total_steps": state.global_step,
                    "status": status,
                }
                if self._critical_failure_detected:
                    summary["critical_failure"] = self._stop_reason
                    summary["last_good_step"] = self._last_good_step
                
                self._experiment_tracker.log_summary(summary)
                
                # Log metrics file as artifact
                if self._artifact_manager and self._artifact_manager.metrics_path.exists():
                    self._experiment_tracker.log_artifact(
                        self._artifact_manager.metrics_path,
                        name="metrics",
                    )
                
                self._experiment_tracker.finish()
                
                if self.verbose:
                    print(f"[DiagnosticsCallback] Experiment tracker finalized")
            except Exception as e:
                if self.verbose:
                    print(f"[DiagnosticsCallback] Tracker finalization failed: {e}")
        
        # Print diagnostics summary at end of run
        if self.auto_diagnostics and self._is_main:
            self._print_diagnostics_summary(state.global_step)

    def _print_diagnostics_summary(self, total_steps: int) -> None:
        """Print a console summary of diagnostics at end of training."""
        import pandas as pd
        
        print(f"\n{'=' * 70}")
        print("POST-TRAINING TOOLKIT - DIAGNOSTICS SUMMARY")
        print(f"{'=' * 70}")
        
        # Basic run info
        print(f"\nTrainer: {self._trainer_type.upper()} | Steps: {total_steps}")
        print(f"Artifacts: {self.run_dir}")
        
        # Distributed training summary
        if self._is_distributed:
            print(f"\n--- Distributed Training ---")
            print(f"  World size: {self._world_size} GPUs")
            
            # Straggler summary
            if self._straggler_detector:
                try:
                    final_report = self._straggler_detector.analyze()
                    if final_report:
                        efficiency = self._straggler_detector.get_efficiency()
                        print(f"  Training efficiency: {efficiency:.1%}")
                        if final_report.has_straggler:
                            print(f"  ⚠️ Straggler: Rank {final_report.slowest_rank} was {final_report.slowdown_factor:.2f}x slower")
                            print(f"     Cause: {final_report.likely_cause}")
                        else:
                            print(f"  ✓ No significant stragglers detected")
                except Exception:
                    pass
            
            # Memory balance summary
            if self._distributed_memory_tracker and self._distributed_memory_tracker.snapshots:
                try:
                    mem_report = self._distributed_memory_tracker.report()
                    if mem_report.current_snapshot.is_imbalanced:
                        print(f"  ⚠️ Memory imbalance: {mem_report.current_snapshot.imbalance_ratio:.1%}")
                        print(f"     Highest: Rank {mem_report.current_snapshot.max_rank} ({mem_report.current_snapshot.max_mb:,.0f} MB)")
                    else:
                        print(f"  ✓ Memory balanced across ranks")
                except Exception:
                    pass
        
        # Profiling summary
        print(f"\n--- Performance Profile ---")
        step_summary = self._step_timer.summary()
        if step_summary.get("total_steps", 0) > 0:
            print(f"  Total time: {step_summary['total_time_sec']:.1f}s")
            print(f"  Mean step time: {step_summary['mean_step_sec']*1000:.0f}ms")
            
            # Check for slowdown
            baseline = step_summary.get("baseline_duration")
            recent = step_summary.get("recent_duration")
            if baseline and recent and recent > baseline * 1.2:
                slowdown_pct = ((recent / baseline) - 1) * 100
                print(f"  ⚠️ Slowdown: {slowdown_pct:.0f}% slower than start")
            else:
                print(f"  ✓ No significant slowdown detected")
                
            # Memory growth
            memory_growth = step_summary.get("memory_growth_mb")
            if memory_growth and memory_growth > 100:
                print(f"  ⚠️ Memory growth: {memory_growth:+,.0f} MB")
        
        # Throughput
        throughput = self._throughput_tracker.report()
        if throughput.mean_tokens_per_sec:
            print(f"  Throughput: {throughput.mean_tokens_per_sec:,.0f} tokens/sec")
        if throughput.bottleneck:
            bottleneck_msg = {
                "io": "⚠️ I/O bound (dataloader bottleneck)",
                "memory": "⚠️ Memory bandwidth limited",
                "compute": "✓ Compute bound (efficient)",
            }.get(throughput.bottleneck, throughput.bottleneck)
            print(f"  {bottleneck_msg}")
        
        # GPU summary
        gpu_report = self._gpu_profiler.report()
        if gpu_report.peak_memory_mb > 0:
            print(f"\n--- GPU Profile ---")
            print(f"  Peak memory: {gpu_report.peak_memory_mb:,.0f} MB")
            print(f"  Memory pressure: {gpu_report.memory_pressure.upper()}")
            if gpu_report.avg_gpu_util is not None:
                print(f"  Avg GPU utilization: {gpu_report.avg_gpu_util:.0f}%")
            if gpu_report.avg_fragmentation > 0.2:
                print(f"  ⚠️ Memory fragmentation: {gpu_report.avg_fragmentation:.0%}")
        
        # Get final metrics from history
        if self._metrics_history:
            df = pd.DataFrame(self._metrics_history)
            
            # Print key final metrics based on trainer type
            print(f"\n--- Training Metrics ---")
            
            if self._trainer_type == "dpo":
                if "dpo_loss" in df.columns:
                    print(f"  DPO Loss: {df['dpo_loss'].iloc[-1]:.4f} (started: {df['dpo_loss'].iloc[0]:.4f})")
                if "win_rate" in df.columns:
                    print(f"  Win Rate: {df['win_rate'].iloc[-1]*100:.1f}% (mean: {df['win_rate'].mean()*100:.1f}%)")
                if "reward_margin" in df.columns:
                    print(f"  Reward Margin: {df['reward_margin'].iloc[-1]:.4f}")
            elif self._trainer_type == "ppo":
                if "ppo_loss" in df.columns:
                    print(f"  PPO Loss: {df['ppo_loss'].iloc[-1]:.4f}")
                if "reward_mean" in df.columns:
                    print(f"  Mean Reward: {df['reward_mean'].iloc[-1]:.4f}")
                if "kl" in df.columns:
                    print(f"  KL Divergence: {df['kl'].iloc[-1]:.4f}")
                if "entropy" in df.columns:
                    print(f"  Entropy: {df['entropy'].iloc[-1]:.4f}")
            elif self._trainer_type == "sft":
                if "sft_loss" in df.columns:
                    print(f"  SFT Loss: {df['sft_loss'].iloc[-1]:.4f} (started: {df['sft_loss'].iloc[0]:.4f})")
            else:
                # Generic: print any loss metric
                loss_cols = [c for c in df.columns if "loss" in c.lower()]
                for col in loss_cols[:3]:
                    print(f"  {col}: {df[col].iloc[-1]:.4f}")
            
            # Run heuristics on full history
            print(f"\n--- Diagnostics ---")
            insights = run_heuristics(df, self._trainer_type)
            
            if not insights:
                print("  ✅ No issues detected")
            else:
                # Group by severity
                high = [i for i in insights if i.severity == "high"]
                medium = [i for i in insights if i.severity == "medium"]
                low = [i for i in insights if i.severity == "low"]
                
                if high:
                    print(f"\n  🚨 HIGH SEVERITY ({len(high)}):")
                    for insight in high:
                        print(f"     • {insight.message}")
                        if insight.reference:
                            print(f"       Ref: {insight.reference}")
                
                if medium:
                    print(f"\n  ⚠️  MEDIUM ({len(medium)}):")
                    for insight in medium:
                        print(f"     • {insight.message}")
                
                if low:
                    print(f"\n  ℹ️  LOW ({len(low)}):")
                    for insight in low[:3]:  # Limit low severity to 3
                        print(f"     • {insight.message}")
                    if len(low) > 3:
                        print(f"     • ... and {len(low) - 3} more")
                
                # Print recommendations
                print(f"\n--- Recommendations ---")
                recommendations = self._get_recommendations(insights)
                for rec in recommendations:
                    print(f"  → {rec}")
        else:
            print("\n  No metrics recorded")
        
        print(f"\n{'=' * 70}\n")
    
    def _get_recommendations(self, insights: List[Insight]) -> List[str]:
        """Generate actionable recommendations from insights and profiling."""
        recommendations = []
        seen_types = set()
        
        # Profiling-based recommendations
        slowdown = self._slowdown_detector.worst_slowdown()
        if slowdown and slowdown.slowdown_factor > 1.5:
            recommendations.append(f"Training slowed {slowdown.slowdown_factor:.1f}x: {slowdown.suggestion}")
        
        gpu_report = self._gpu_profiler.report()
        if gpu_report.memory_pressure == "high":
            recommendations.append("High GPU memory pressure: enable gradient_checkpointing or reduce batch size")
        if gpu_report.avg_fragmentation > 0.3:
            recommendations.append("High memory fragmentation: restart process between runs or use torch.cuda.empty_cache()")
        
        throughput = self._throughput_tracker.report()
        if throughput.bottleneck == "io":
            recommendations.append("I/O bottleneck: increase dataloader num_workers or use pin_memory=True")
        
        for insight in insights:
            if insight.type in seen_types:
                continue
            seen_types.add(insight.type)
            
            # DPO recommendations
            if insight.type == "dpo_loss_random":
                recommendations.append("DPO loss at random chance: increase learning rate 2-5x, check data quality, or reduce beta")
            elif insight.type == "margin_collapse":
                recommendations.append("Reward margin collapsing: reduce learning rate or increase KL penalty")
            elif insight.type == "win_rate_unstable":
                recommendations.append("Win rate unstable: increase batch size for more stable gradients")
            
            # PPO recommendations
            elif insight.type == "entropy_collapse":
                recommendations.append("Entropy collapse: increase entropy coefficient or reduce learning rate")
            elif insight.type == "value_head_divergence":
                recommendations.append("Value head diverging: clip value function updates or reduce vf_coef")
            elif insight.type == "kl_instability":
                recommendations.append("KL too high: reduce learning rate or increase KL penalty")
            elif insight.type == "advantage_explosion":
                recommendations.append("Advantages exploding: normalize advantages or clip gradients")
            
            # SFT recommendations
            elif insight.type == "loss_plateau":
                recommendations.append("Loss plateaued: try learning rate warmup restart or check for data issues")
            elif insight.type == "perplexity_spike":
                recommendations.append("Perplexity spiked: check for bad batches or reduce learning rate")
            
            # GRPO recommendations
            elif insight.type == "grpo_entropy_collapse":
                recommendations.append("GRPO entropy collapse: increase temperature or entropy bonus")
            
            # Generic
            elif insight.type == "reward_variance_spike":
                recommendations.append("Reward variance spiked: check for reward hacking or distribution shift")
            elif insight.type == "nan_detected":
                recommendations.append("NaN detected: reduce learning rate, check for numerical instability")
        
        if not recommendations:
            recommendations.append("Training looks healthy. Review metrics history for fine-tuning.")
        
        return recommendations[:5]  # Limit to top 5

    def capture_snapshot(self, step: int, model: Any, tokenizer: Any) -> None:
        """Manually capture a behavior snapshot.
        
        Use this for explicit snapshot capture outside the automatic interval.
        
        Args:
            step: Current step number
            model: Model to evaluate
            tokenizer: Tokenizer for the model
        """
        if self._snapshot_manager:
            self._snapshot_manager.capture(step, model, tokenizer)

    @property
    def trainer_type(self) -> str:
        """Return the detected trainer type."""
        return self._trainer_type
    
    @property
    def artifact_manager(self) -> Optional[RunArtifactManager]:
        """Return the artifact manager for advanced usage."""
        return self._artifact_manager
    
    @property
    def snapshot_manager(self) -> Optional[SnapshotManager]:
        """Return the snapshot manager for advanced usage."""
        return self._snapshot_manager
