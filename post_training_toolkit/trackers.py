"""Experiment tracker integrations for WandB, MLflow, and TensorBoard.

This module provides optional integrations with popular experiment tracking
platforms. All trackers are lazy-loaded to avoid import overhead when not used.

Usage:
    from post_training_toolkit.trackers import get_tracker
    
    # WandB
    tracker = get_tracker("wandb", project="my-project", name="run-1")
    tracker.log_metrics({"loss": 0.5}, step=100)
    
    # MLflow
    tracker = get_tracker("mlflow", experiment_name="my-experiment")
    tracker.log_metrics({"loss": 0.5}, step=100)
    
    # TensorBoard
    tracker = get_tracker("tensorboard", log_dir="runs/my-run")
    tracker.log_metrics({"loss": 0.5}, step=100)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pathlib import Path
import warnings


class ExperimentTracker(ABC):
    """Abstract base class for experiment trackers."""
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics at a given step."""
        pass
    
    @abstractmethod
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration/hyperparameters."""
        pass
    
    @abstractmethod
    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        """Log an artifact (file)."""
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Finalize the tracker."""
        pass
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log summary metrics (called at end of training)."""
        # Default implementation - subclasses may override
        self.log_metrics(summary, step=-1)


class WandBTracker(ExperimentTracker):
    """Weights & Biases experiment tracker.
    
    Requires: pip install wandb
    """
    
    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        **kwargs,
    ):
        """Initialize WandB tracker.
        
        Args:
            project: WandB project name
            name: Run name (auto-generated if None)
            config: Initial config to log
            tags: Tags for the run
            notes: Notes for the run
            **kwargs: Additional arguments passed to wandb.init()
        """
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "WandB not installed. Install with: pip install wandb\n"
                "Or: pip install post-training-toolkit[wandb]"
            )
        
        self._run = self._wandb.init(
            project=project or "post-training-toolkit",
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            **kwargs,
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to WandB."""
        self._wandb.log(metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Update WandB config."""
        self._wandb.config.update(config)
    
    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        """Log artifact to WandB."""
        artifact = self._wandb.Artifact(
            name=name or Path(path).stem,
            type="diagnostics",
        )
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log summary to WandB."""
        for key, value in summary.items():
            self._wandb.run.summary[key] = value
    
    def finish(self) -> None:
        """Finish WandB run."""
        self._wandb.finish()


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker.
    
    Requires: pip install mlflow
    """
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Initialize MLflow tracker.
        
        Args:
            experiment_name: MLflow experiment name
            run_name: Run name
            tracking_uri: MLflow tracking server URI
            tags: Tags for the run
            **kwargs: Additional arguments
        """
        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            raise ImportError(
                "MLflow not installed. Install with: pip install mlflow\n"
                "Or: pip install post-training-toolkit[mlflow]"
            )
        
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)
        
        if experiment_name:
            self._mlflow.set_experiment(experiment_name)
        
        self._run = self._mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to MLflow."""
        self._mlflow.log_metrics(metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log params to MLflow."""
        # MLflow params must be strings
        flat_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    flat_config[f"{key}.{subkey}"] = str(subval)
            else:
                flat_config[key] = str(value)
        self._mlflow.log_params(flat_config)
    
    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        """Log artifact to MLflow."""
        self._mlflow.log_artifact(str(path))
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log summary metrics to MLflow."""
        # Log as final metrics
        numeric_summary = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        if numeric_summary:
            self._mlflow.log_metrics(numeric_summary)
    
    def finish(self) -> None:
        """End MLflow run."""
        self._mlflow.end_run()


class TensorBoardTracker(ExperimentTracker):
    """TensorBoard experiment tracker.
    
    Requires: pip install tensorboard
    """
    
    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        comment: Optional[str] = None,
        **kwargs,
    ):
        """Initialize TensorBoard tracker.
        
        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment appended to log directory name
            **kwargs: Additional arguments passed to SummaryWriter
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._SummaryWriter = SummaryWriter
        except ImportError:
            raise ImportError(
                "TensorBoard not installed. Install with: pip install tensorboard\n"
                "Or: pip install post-training-toolkit[tensorboard]"
            )
        
        self._writer = self._SummaryWriter(
            log_dir=str(log_dir) if log_dir else None,
            comment=comment,
            **kwargs,
        )
        self._config: Dict[str, Any] = {}
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to TensorBoard."""
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Store config (TensorBoard doesn't natively support config)."""
        self._config.update(config)
        # Log as text
        import json
        self._writer.add_text("config", json.dumps(config, indent=2))
    
    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        """TensorBoard doesn't support artifacts - log as text reference."""
        self._writer.add_text(
            f"artifact/{name or Path(path).name}",
            f"Artifact saved to: {path}",
        )
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log summary as hparams."""
        try:
            self._writer.add_hparams(
                hparam_dict=self._config,
                metric_dict={k: v for k, v in summary.items() if isinstance(v, (int, float))},
            )
        except Exception:
            # Fallback to text
            import json
            self._writer.add_text("summary", json.dumps(summary, indent=2))
    
    def finish(self) -> None:
        """Close TensorBoard writer."""
        self._writer.close()


class NoOpTracker(ExperimentTracker):
    """No-op tracker for when tracking is disabled."""
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass
    
    def log_config(self, config: Dict[str, Any]) -> None:
        pass
    
    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        pass
    
    def finish(self) -> None:
        pass


def get_tracker(
    tracker_type: Optional[str] = None,
    **kwargs,
) -> ExperimentTracker:
    """Get an experiment tracker by type.
    
    Args:
        tracker_type: One of "wandb", "mlflow", "tensorboard", or None
        **kwargs: Arguments passed to the tracker constructor
        
    Returns:
        ExperimentTracker instance
        
    Raises:
        ValueError: If tracker_type is not recognized
        ImportError: If required package is not installed
    """
    if tracker_type is None:
        return NoOpTracker()
    
    tracker_type = tracker_type.lower()
    
    if tracker_type == "wandb":
        return WandBTracker(**kwargs)
    elif tracker_type == "mlflow":
        return MLflowTracker(**kwargs)
    elif tracker_type in ("tensorboard", "tb"):
        return TensorBoardTracker(**kwargs)
    else:
        raise ValueError(
            f"Unknown tracker type: {tracker_type}. "
            f"Supported: wandb, mlflow, tensorboard"
        )


# Convenience function for auto-detection
def auto_detect_tracker() -> Optional[str]:
    """Auto-detect available tracker based on installed packages.
    
    Returns the first available tracker type, or None if none available.
    Priority: wandb > mlflow > tensorboard
    """
    try:
        import wandb
        return "wandb"
    except ImportError:
        pass
    
    try:
        import mlflow
        return "mlflow"
    except ImportError:
        pass
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        return "tensorboard"
    except ImportError:
        pass
    
    return None
