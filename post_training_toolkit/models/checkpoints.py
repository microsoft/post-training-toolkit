"""Checkpoint comparison and selection for post-training runs.

Compares checkpoints using:
- Metric stability (loss variance, reward consistency)
- Behavioral drift signals (from snapshots)
- Length and entropy consistency
- Refusal rate

Outputs a recommended checkpoint with justification.
This is decision support, not an oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from post_training_toolkit.models.artifacts import RunArtifactManager, Snapshot
from post_training_toolkit.models.diffing import DiffManager, DiffSummary, diff_snapshots


@dataclass
class CheckpointScore:
    """Scores for a single checkpoint."""
    step: int
    stability_score: float  # Higher = more stable metrics
    drift_score: float  # Lower = less drift from baseline
    refusal_rate: float
    length_consistency: float  # Higher = more consistent lengths
    entropy_consistency: float  # Higher = more consistent entropy
    overall_score: float  # Weighted combination
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "stability": self.stability_score,
            "drift": self.drift_score,
            "refusal_rate": self.refusal_rate,
            "length_consistency": self.length_consistency,
            "entropy_consistency": self.entropy_consistency,
            "overall": self.overall_score,
        }


@dataclass
class CheckpointRecommendation:
    """Checkpoint selection recommendation."""
    step: int
    justification: str
    candidates: List[CheckpointScore]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "justification": self.justification,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def compute_metric_stability(
    df: pd.DataFrame,
    step: int,
    window: int = 10,
    loss_key: str = "loss",
) -> float:
    """Compute metric stability score around a step.
    
    Higher score = more stable metrics.
    
    Args:
        df: DataFrame with metrics
        step: Step to evaluate
        window: Window size for stability calculation
        loss_key: Key for loss metric
        
    Returns:
        Stability score (0-1)
    """
    # Get window around step
    mask = (df["step"] >= step - window) & (df["step"] <= step + window)
    window_df = df[mask]
    
    if len(window_df) < 3:
        return 0.5  # Not enough data
    
    scores = []
    
    # Loss stability (lower variance = better)
    for key in [loss_key, "dpo_loss", "ppo_loss", "sft_loss"]:
        if key in window_df.columns:
            variance = window_df[key].var()
            # Convert to 0-1 score (lower variance = higher score)
            stability = 1.0 / (1.0 + variance * 10)
            scores.append(stability)
            break
    
    # Reward stability
    if "reward_mean" in window_df.columns:
        variance = window_df["reward_mean"].var()
        stability = 1.0 / (1.0 + variance * 5)
        scores.append(stability)
    
    # KL stability (if available)
    if "kl" in window_df.columns:
        variance = window_df["kl"].var()
        stability = 1.0 / (1.0 + variance * 2)
        scores.append(stability)
    
    if not scores:
        return 0.5
    
    return sum(scores) / len(scores)


def compute_drift_score(
    baseline_snapshot: Snapshot,
    current_snapshot: Snapshot,
) -> Tuple[float, DiffSummary]:
    """Compute drift score from baseline.
    
    Lower score = less drift (better).
    
    Args:
        baseline_snapshot: Baseline snapshot (usually step 0)
        current_snapshot: Current snapshot to evaluate
        
    Returns:
        Tuple of (drift_score, DiffSummary)
    """
    _, summary = diff_snapshots(baseline_snapshot, current_snapshot)
    
    # Compute drift score from summary
    # Weight different factors
    drift = 0.0
    
    # Refusal changes are heavily penalized
    refusal_delta = abs(summary.refusal_rate_after - summary.refusal_rate_before)
    drift += refusal_delta * 3.0
    
    # Length changes
    total = summary.length_increased + summary.length_decreased + summary.length_stable
    if total > 0:
        length_change_rate = (summary.length_increased + summary.length_decreased) / total
        drift += length_change_rate * 0.5
    
    # Significant changes
    if summary.num_prompts > 0:
        significant_rate = summary.significant_changes / summary.num_prompts
        drift += significant_rate * 1.0
    
    return drift, summary


def compute_snapshot_consistency(snapshot: Snapshot) -> Tuple[float, float]:
    """Compute length and entropy consistency within a snapshot.
    
    Higher scores = more consistent.
    
    Args:
        snapshot: Snapshot to analyze
        
    Returns:
        Tuple of (length_consistency, entropy_consistency)
    """
    if not snapshot.entries:
        return 0.5, 0.5
    
    # Length consistency (inverse of coefficient of variation)
    lengths = [e.output_length for e in snapshot.entries]
    mean_len = sum(lengths) / len(lengths)
    if mean_len > 0:
        std_len = (sum((x - mean_len) ** 2 for x in lengths) / len(lengths)) ** 0.5
        cv_len = std_len / mean_len
        length_consistency = 1.0 / (1.0 + cv_len)
    else:
        length_consistency = 0.5
    
    # Entropy consistency
    entropies = [e.entropy_mean for e in snapshot.entries if e.entropy_mean is not None]
    if entropies:
        mean_ent = sum(entropies) / len(entropies)
        if mean_ent > 0:
            std_ent = (sum((x - mean_ent) ** 2 for x in entropies) / len(entropies)) ** 0.5
            cv_ent = std_ent / mean_ent
            entropy_consistency = 1.0 / (1.0 + cv_ent)
        else:
            entropy_consistency = 0.5
    else:
        entropy_consistency = 0.5
    
    return length_consistency, entropy_consistency


class CheckpointComparator:
    """Compare checkpoints and recommend the best one."""
    
    def __init__(
        self,
        artifact_manager: RunArtifactManager,
        stability_weight: float = 0.3,
        drift_weight: float = 0.4,
        consistency_weight: float = 0.2,
        refusal_weight: float = 0.1,
    ):
        """Initialize checkpoint comparator.
        
        Args:
            artifact_manager: Manages run artifacts
            stability_weight: Weight for metric stability
            drift_weight: Weight for behavioral drift
            consistency_weight: Weight for output consistency
            refusal_weight: Weight for refusal rate
        """
        self.artifact_manager = artifact_manager
        self.stability_weight = stability_weight
        self.drift_weight = drift_weight
        self.consistency_weight = consistency_weight
        self.refusal_weight = refusal_weight
        self._df: Optional[pd.DataFrame] = None
    
    def load_metrics(self) -> pd.DataFrame:
        """Load metrics from artifact manager."""
        if self._df is not None:
            return self._df
        
        import json
        
        records = []
        metrics_path = self.artifact_manager.metrics_path
        
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    if obj.get("type") in ("header", "footer"):
                        continue
                    step = obj.get("step")
                    metrics = obj.get("metrics", {})
                    record = {"step": step}
                    record.update(metrics)
                    records.append(record)
        
        self._df = pd.DataFrame.from_records(records)
        if len(self._df) > 0:
            self._df = self._df.sort_values("step").reset_index(drop=True)
        
        return self._df
    
    def score_checkpoint(
        self,
        step: int,
        baseline_snapshot: Optional[Snapshot] = None,
    ) -> Optional[CheckpointScore]:
        """Score a single checkpoint.
        
        Args:
            step: Step to score
            baseline_snapshot: Baseline for drift comparison
            
        Returns:
            CheckpointScore or None if insufficient data
        """
        df = self.load_metrics()
        snapshot = self.artifact_manager.load_snapshot(step)
        
        if snapshot is None:
            return None
        
        # Metric stability
        stability = compute_metric_stability(df, step)
        
        # Drift from baseline
        drift = 0.0
        if baseline_snapshot is not None:
            drift, _ = compute_drift_score(baseline_snapshot, snapshot)
        
        # Consistency
        length_cons, entropy_cons = compute_snapshot_consistency(snapshot)
        
        # Refusal rate
        refusal_rate = sum(1 for e in snapshot.entries if e.is_refusal) / max(len(snapshot.entries), 1)
        
        # Overall score (higher = better)
        # Note: drift is inverted (lower drift = higher score contribution)
        overall = (
            self.stability_weight * stability +
            self.drift_weight * (1.0 - min(drift, 1.0)) +
            self.consistency_weight * (length_cons + entropy_cons) / 2 +
            self.refusal_weight * (1.0 - refusal_rate)
        )
        
        return CheckpointScore(
            step=step,
            stability_score=stability,
            drift_score=drift,
            refusal_rate=refusal_rate,
            length_consistency=length_cons,
            entropy_consistency=entropy_cons,
            overall_score=overall,
        )
    
    def compare_checkpoints(self) -> Optional[CheckpointRecommendation]:
        """Compare all available checkpoints and recommend the best.
        
        Returns:
            CheckpointRecommendation or None if insufficient data
        """
        steps = self.artifact_manager.list_snapshots()
        
        if len(steps) < 2:
            return None
        
        # Use first snapshot as baseline
        baseline = self.artifact_manager.load_snapshot(steps[0])
        
        # Score all checkpoints
        scores: List[CheckpointScore] = []
        for step in steps:
            score = self.score_checkpoint(step, baseline)
            if score:
                scores.append(score)
        
        if not scores:
            return None
        
        # Sort by overall score (descending)
        scores.sort(key=lambda s: s.overall_score, reverse=True)
        
        best = scores[0]
        
        # Generate justification
        justification_parts = []
        
        if best.stability_score > 0.7:
            justification_parts.append("stable metrics")
        if best.drift_score < 0.3:
            justification_parts.append("low behavioral drift")
        if best.refusal_rate < 0.1:
            justification_parts.append("low refusal rate")
        if best.length_consistency > 0.7:
            justification_parts.append("consistent output lengths")
        
        if justification_parts:
            justification = f"Step {best.step} is recommended due to: {', '.join(justification_parts)}."
        else:
            justification = f"Step {best.step} has the best overall score among available checkpoints."
        
        # Add warnings if needed
        if best.refusal_rate > 0.2:
            justification += f" Note: refusal rate ({best.refusal_rate:.1%}) is elevated."
        if best.drift_score > 0.5:
            justification += f" Note: significant drift from baseline detected."
        
        return CheckpointRecommendation(
            step=best.step,
            justification=justification,
            candidates=scores,
        )


def recommend_checkpoint(run_dir: Path | str) -> Optional[CheckpointRecommendation]:
    """Convenience function to get checkpoint recommendation for a run.
    
    Args:
        run_dir: Path to run artifacts directory
        
    Returns:
        CheckpointRecommendation or None
    """
    artifact_manager = RunArtifactManager(Path(run_dir))
    comparator = CheckpointComparator(artifact_manager)
    return comparator.compare_checkpoints()


# =============================================================================
# Resume Validation
# =============================================================================

@dataclass
class ResumeValidationResult:
    """Result of resume validation check."""
    is_valid: bool
    resumed_from_step: int
    expected_next_step: int
    actual_next_step: Optional[int]
    warnings: List[str]
    errors: List[str]
    checkpoint_path: Optional[str] = None
    state_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "resumed_from_step": self.resumed_from_step,
            "expected_next_step": self.expected_next_step,
            "actual_next_step": self.actual_next_step,
            "warnings": self.warnings,
            "errors": self.errors,
            "checkpoint_path": self.checkpoint_path,
            "state_hash": self.state_hash,
        }


class ResumeValidator:
    """Validates that a resumed training run is correctly continuing.
    
    Checks:
    - Step continuity (resumed step matches checkpoint)
    - No gap or overlap in step sequence
    - Metrics file continuity
    - Optional: config consistency
    
    Usage:
        validator = ResumeValidator(artifact_manager)
        
        # On resume, call this with the checkpoint you're loading from
        result = validator.validate_resume(
            checkpoint_step=1000,
            checkpoint_path="/path/to/checkpoint-1000",
        )
        
        if not result.is_valid:
            print(f"Resume validation failed: {result.errors}")
        
        # After first step of resumed training
        validator.verify_first_step(step=1001)
    """
    
    def __init__(self, artifact_manager: RunArtifactManager):
        """Initialize resume validator.
        
        Args:
            artifact_manager: Manages run artifacts
        """
        self.artifact_manager = artifact_manager
        self._checkpoint_step: Optional[int] = None
        self._checkpoint_path: Optional[str] = None
        self._expected_next_step: Optional[int] = None
    
    def validate_resume(
        self,
        checkpoint_step: int,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ResumeValidationResult:
        """Validate that resume from checkpoint is correct.
        
        Should be called after loading checkpoint but before resuming training.
        
        Args:
            checkpoint_step: Step number of checkpoint being resumed from
            checkpoint_path: Optional path to checkpoint directory
            config: Optional training config to compare with original
            
        Returns:
            ResumeValidationResult with validation status
        """
        warnings = []
        errors = []
        
        # Store for later verification
        self._checkpoint_step = checkpoint_step
        self._checkpoint_path = checkpoint_path
        self._expected_next_step = checkpoint_step + 1
        
        # Check if we have metrics history
        last_logged_step = self._get_last_logged_step()
        
        if last_logged_step is not None:
            # Validate step continuity
            if checkpoint_step > last_logged_step:
                errors.append(
                    f"Checkpoint step {checkpoint_step} is ahead of last logged step {last_logged_step}. "
                    f"Metrics may be missing."
                )
            elif checkpoint_step < last_logged_step:
                warnings.append(
                    f"Resuming from step {checkpoint_step} but metrics exist up to step {last_logged_step}. "
                    f"This may indicate a previous crash after the checkpoint was saved."
                )
        
        # Check if snapshot exists for checkpoint step
        snapshots = self.artifact_manager.list_snapshots()
        if snapshots and checkpoint_step not in snapshots:
            # Find closest snapshot
            closest = min(snapshots, key=lambda s: abs(s - checkpoint_step))
            warnings.append(
                f"No snapshot at checkpoint step {checkpoint_step}. Closest snapshot at step {closest}."
            )
        
        # Config validation (if original config available)
        if config is not None:
            config_warnings = self._validate_config(config)
            warnings.extend(config_warnings)
        
        # Compute state hash if checkpoint path provided
        state_hash = None
        if checkpoint_path:
            state_hash = self._compute_checkpoint_hash(checkpoint_path)
        
        is_valid = len(errors) == 0
        
        return ResumeValidationResult(
            is_valid=is_valid,
            resumed_from_step=checkpoint_step,
            expected_next_step=self._expected_next_step,
            actual_next_step=None,  # Will be set by verify_first_step
            warnings=warnings,
            errors=errors,
            checkpoint_path=checkpoint_path,
            state_hash=state_hash,
        )
    
    def verify_first_step(self, step: int) -> Tuple[bool, List[str]]:
        """Verify the first step after resume is correct.
        
        Call this after the first training step completes.
        
        Args:
            step: The step number of the first step after resume
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if self._expected_next_step is None:
            errors.append("verify_first_step called before validate_resume")
            return False, errors
        
        if step != self._expected_next_step:
            errors.append(
                f"First step after resume is {step}, expected {self._expected_next_step}. "
                f"This indicates a step counter mismatch."
            )
        
        # Check for duplicate step in metrics
        if self._has_duplicate_step(step):
            errors.append(
                f"Step {step} already exists in metrics log. "
                f"Resume may have caused duplicate logging."
            )
        
        return len(errors) == 0, errors
    
    def _get_last_logged_step(self) -> Optional[int]:
        """Get the last step logged to metrics file."""
        import json
        
        metrics_path = self.artifact_manager.metrics_path
        if not metrics_path.exists():
            return None
        
        last_step = None
        with open(metrics_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "step" in obj:
                        last_step = obj["step"]
                except json.JSONDecodeError:
                    continue
        
        return last_step
    
    def _has_duplicate_step(self, step: int) -> bool:
        """Check if a step already exists in metrics."""
        import json
        
        metrics_path = self.artifact_manager.metrics_path
        if not metrics_path.exists():
            return False
        
        with open(metrics_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("step") == step:
                        return True
                except json.JSONDecodeError:
                    continue
        
        return False
    
    def _validate_config(self, new_config: Dict[str, Any]) -> List[str]:
        """Compare new config with original run config.
        
        Returns list of warnings for mismatched critical parameters.
        """
        warnings = []
        
        # Load original metadata
        import json
        
        metadata_path = self.artifact_manager.metadata_path
        if not metadata_path.exists():
            return warnings
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            original_config = metadata.get("config", {})
        except (json.JSONDecodeError, KeyError):
            return warnings
        
        # Critical params to check
        critical_params = [
            "learning_rate", "lr", 
            "batch_size", "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "beta",  # DPO
            "max_length", "max_seq_length",
        ]
        
        for param in critical_params:
            orig_val = original_config.get(param)
            new_val = new_config.get(param)
            
            if orig_val is not None and new_val is not None and orig_val != new_val:
                warnings.append(
                    f"Config mismatch: {param} changed from {orig_val} to {new_val}"
                )
        
        return warnings
    
    def _compute_checkpoint_hash(self, checkpoint_path: str) -> Optional[str]:
        """Compute a hash of checkpoint files for integrity verification."""
        import hashlib
        
        try:
            ckpt_dir = Path(checkpoint_path)
            if not ckpt_dir.exists():
                return None
            
            # Hash key files
            hasher = hashlib.sha256()
            
            key_files = [
                "trainer_state.json",
                "optimizer.pt", 
                "scheduler.pt",
                "adapter_config.json",
            ]
            
            for filename in key_files:
                filepath = ckpt_dir / filename
                if filepath.exists():
                    # Hash file size and first 1KB (fast approximation)
                    stat = filepath.stat()
                    hasher.update(f"{filename}:{stat.st_size}:".encode())
                    with open(filepath, "rb") as f:
                        hasher.update(f.read(1024))
            
            return hasher.hexdigest()[:16]
        except Exception:
            return None


def validate_resume(
    run_dir: Path | str,
    checkpoint_step: int,
    checkpoint_path: Optional[str] = None,
) -> ResumeValidationResult:
    """Convenience function to validate a resume.
    
    Args:
        run_dir: Path to run artifacts directory
        checkpoint_step: Step number being resumed from
        checkpoint_path: Optional path to checkpoint directory
        
    Returns:
        ResumeValidationResult
    """
    artifact_manager = RunArtifactManager(Path(run_dir))
    validator = ResumeValidator(artifact_manager)
    return validator.validate_resume(checkpoint_step, checkpoint_path)
