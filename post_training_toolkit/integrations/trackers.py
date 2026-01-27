from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pathlib import Path
import warnings

class ExperimentTracker(ABC):
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass
    
    @abstractmethod
    def log_config(self, config: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        pass
    
    @abstractmethod
    def finish(self) -> None:
        pass
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        self.log_metrics(summary, step=-1)

class WandBTracker(ExperimentTracker):
    
    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        **kwargs,
    ):
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
        self._wandb.log(metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        self._wandb.config.update(config)
    
    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        artifact = self._wandb.Artifact(
            name=name or Path(path).stem,
            type="diagnostics",
        )
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        for key, value in summary.items():
            self._wandb.run.summary[key] = value
    
    def finish(self) -> None:
        self._wandb.finish()

class MLflowTracker(ExperimentTracker):
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
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
        self._mlflow.log_metrics(metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        flat_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    flat_config[f"{key}.{subkey}"] = str(subval)
            else:
                flat_config[key] = str(value)
        self._mlflow.log_params(flat_config)
    
    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        self._mlflow.log_artifact(str(path))
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        numeric_summary = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        if numeric_summary:
            self._mlflow.log_metrics(numeric_summary)
    
    def finish(self) -> None:
        self._mlflow.end_run()

class TensorBoardTracker(ExperimentTracker):
    
    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        comment: Optional[str] = None,
        **kwargs,
    ):
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
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        self._config.update(config)
        import json
        self._writer.add_text("config", json.dumps(config, indent=2))
    
    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        self._writer.add_text(
            f"artifact/{name or Path(path).name}",
            f"Artifact saved to: {path}",
        )
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        try:
            self._writer.add_hparams(
                hparam_dict=self._config,
                metric_dict={k: v for k, v in summary.items() if isinstance(v, (int, float))},
            )
        except Exception:
            import json
            self._writer.add_text("summary", json.dumps(summary, indent=2))
    
    def finish(self) -> None:
        self._writer.close()

class NoOpTracker(ExperimentTracker):
    
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

def auto_detect_tracker() -> Optional[str]:
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
