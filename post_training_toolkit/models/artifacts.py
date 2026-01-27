from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ARTIFACT_SCHEMA_VERSION = "1.0"

@dataclass
class RunMetadata:
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    run_id: str = ""
    trainer_type: str = "unknown"
    
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    ref_model_name: Optional[str] = None
    ref_model_revision: Optional[str] = None
    tokenizer_name: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    
    dataset_name: Optional[str] = None
    dataset_fingerprint: Optional[str] = None
    dataset_size: Optional[int] = None
    
    start_time: str = ""
    end_time: Optional[str] = None
    status: str = "running"
    total_steps: int = 0
    
    world_size: int = 1
    global_rank: int = 0
    
    git_sha: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    git_diff_path: Optional[str] = None
    
    config: Dict[str, Any] = field(default_factory=dict)
    config_hash: Optional[str] = None
    
    environment: Dict[str, Any] = field(default_factory=dict)
    package_versions: Dict[str, str] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    
    git_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunMetadata":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

@dataclass 
class SnapshotMetadata:
    step: int
    timestamp: str
    num_prompts: int
    generation_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SnapshotEntry:
    prompt_id: str
    prompt: str
    output: str
    output_length: int
    is_refusal: bool
    logprob_mean: Optional[float] = None
    logprob_std: Optional[float] = None
    entropy_mean: Optional[float] = None
    entropy_std: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SnapshotEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

@dataclass
class Snapshot:
    metadata: SnapshotMetadata
    entries: List[SnapshotEntry]
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "entries": [e.to_dict() for e in self.entries],
            "summary": self.summary,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Snapshot":
        return cls(
            metadata=SnapshotMetadata(**d["metadata"]),
            entries=[SnapshotEntry.from_dict(e) for e in d["entries"]],
            summary=d.get("summary", {}),
        )

@dataclass
class DiffEntry:
    prompt_id: str
    length_delta: int
    length_pct_change: float
    refusal_changed: bool
    refusal_before: bool
    refusal_after: bool
    entropy_delta: Optional[float] = None
    logprob_delta: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SnapshotDiff:
    step_a: int
    step_b: int
    timestamp: str
    entries: List[DiffEntry]
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_a": self.step_a,
            "step_b": self.step_b,
            "timestamp": self.timestamp,
            "entries": [e.to_dict() for e in self.entries],
            "summary": self.summary,
        }

@dataclass
class Postmortem:
    exit_reason: str
    exit_code: Optional[int] = None
    last_step: int = 0
    timestamp: str = ""
    traceback: Optional[str] = None
    last_metrics: Dict[str, Any] = field(default_factory=dict)
    recent_events: List[str] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    cuda_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def get_environment_info() -> Dict[str, Any]:
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
    }
    
    try:
        import torch
        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["gpu_count"] = torch.cuda.device_count()
            env_info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    
    try:
        import transformers
        env_info["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    
    try:
        import trl
        env_info["trl_version"] = trl.__version__
    except ImportError:
        pass
        
    return env_info

def get_git_info(repo_path: Optional[Path] = None, save_diff: bool = False, 
                 diff_dir: Optional[Path] = None) -> Dict[str, Any]:
    import subprocess
    
    git_info: Dict[str, Any] = {
        "git_sha": None,
        "git_branch": None,
        "git_dirty": False,
        "git_diff_path": None,
    }
    
    try:
        cwd = str(repo_path) if repo_path else None
        
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=cwd, timeout=5
        )
        if result.returncode == 0:
            git_info["git_sha"] = result.stdout.strip()
        
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=cwd, timeout=5
        )
        if result.returncode == 0:
            git_info["git_branch"] = result.stdout.strip()
        
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=cwd, timeout=5
        )
        if result.returncode == 0:
            git_info["git_dirty"] = len(result.stdout.strip()) > 0
            
            if save_diff and git_info["git_dirty"] and diff_dir:
                diff_result = subprocess.run(
                    ["git", "diff", "HEAD"],
                    capture_output=True, text=True, cwd=cwd, timeout=30
                )
                if diff_result.returncode == 0 and diff_result.stdout:
                    diff_path = diff_dir / "uncommitted_changes.diff"
                    diff_path.write_text(diff_result.stdout)
                    git_info["git_diff_path"] = str(diff_path)
                    
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return git_info

def get_hardware_info() -> Dict[str, Any]:
    import subprocess
    
    hw_info: Dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version,
    }
    
    try:
        import torch
        hw_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw_info["cuda_version"] = torch.version.cuda
            hw_info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
            hw_info["gpu_count"] = torch.cuda.device_count()
            hw_info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            hw_info["gpu_memory_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        hw_info["cuda_available"] = False
    
    hw_info["world_size"] = get_world_size()
    hw_info["global_rank"] = get_global_rank()
    hw_info["local_rank"] = int(os.environ.get("LOCAL_RANK", "0"))
    
    return hw_info

def get_package_versions(full_env: bool = False) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    
    core_packages = [
        "torch", "transformers", "trl", "datasets", "tokenizers",
        "accelerate", "peft", "bitsandbytes", "safetensors",
        "numpy", "pandas", "scipy", "scikit-learn",
    ]
    
    for pkg in core_packages:
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            pass
    
    if full_env:
        try:
            import importlib.metadata
            for dist in importlib.metadata.distributions():
                name = dist.metadata.get("Name", "")
                version = dist.metadata.get("Version", "unknown")
                if name and name not in versions:
                    versions[name] = version
        except Exception:
            pass
    
    return versions

def compute_config_hash(config: Dict[str, Any]) -> str:
    import hashlib
    
    def _normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _normalize(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, (list, tuple)):
            return [_normalize(x) for x in obj]
        elif isinstance(obj, float):
            return round(obj, 10)
        elif hasattr(obj, "__dict__"):
            return _normalize(obj.__dict__)
        else:
            return obj
    
    normalized = _normalize(config)
    config_str = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]

def get_model_identity(model: Any) -> Dict[str, Any]:
    identity: Dict[str, Any] = {
        "model_name": None,
        "model_revision": None,
    }
    
    if isinstance(model, str):
        identity["model_name"] = model
        return identity
    
    if hasattr(model, "name_or_path"):
        identity["model_name"] = model.name_or_path
    elif hasattr(model, "config"):
        config = model.config
        if hasattr(config, "_name_or_path"):
            identity["model_name"] = config._name_or_path
        if hasattr(config, "_commit_hash"):
            identity["model_revision"] = config._commit_hash
    
    return identity

def get_tokenizer_identity(tokenizer: Any) -> Dict[str, Any]:
    identity: Dict[str, Any] = {
        "tokenizer_name": None,
        "tokenizer_revision": None,
    }
    
    if isinstance(tokenizer, str):
        identity["tokenizer_name"] = tokenizer
        return identity
    
    if hasattr(tokenizer, "name_or_path"):
        identity["tokenizer_name"] = tokenizer.name_or_path
    
    return identity

def get_dataset_identity(dataset: Any) -> Dict[str, Any]:
    identity: Dict[str, Any] = {
        "dataset_name": None,
        "dataset_fingerprint": None,
        "dataset_size": None,
    }
    
    if dataset is None:
        return identity
    
    if hasattr(dataset, "_fingerprint"):
        identity["dataset_fingerprint"] = dataset._fingerprint
    
    if hasattr(dataset, "info") and dataset.info:
        info = dataset.info
        if hasattr(info, "dataset_name"):
            identity["dataset_name"] = info.dataset_name
    
    try:
        if hasattr(dataset, "__len__"):
            identity["dataset_size"] = len(dataset)
        elif hasattr(dataset, "num_rows"):
            identity["dataset_size"] = dataset.num_rows
    except Exception:
        pass
    
    return identity

def collect_full_provenance(
    model: Any = None,
    tokenizer: Any = None,
    dataset: Any = None,
    config: Optional[Dict[str, Any]] = None,
    repo_path: Optional[Path] = None,
    save_git_diff: bool = False,
    diff_dir: Optional[Path] = None,
    full_package_snapshot: bool = False,
) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {}
    
    git_info = get_git_info(repo_path, save_git_diff, diff_dir)
    provenance.update(git_info)
    
    provenance["hardware"] = get_hardware_info()
    
    provenance["package_versions"] = get_package_versions(full_package_snapshot)
    
    if model is not None:
        model_id = get_model_identity(model)
        provenance["model_name"] = model_id["model_name"]
        provenance["model_revision"] = model_id["model_revision"]
    
    if tokenizer is not None:
        tok_id = get_tokenizer_identity(tokenizer)
        provenance["tokenizer_name"] = tok_id["tokenizer_name"]
        provenance["tokenizer_revision"] = tok_id["tokenizer_revision"]
    
    if dataset is not None:
        ds_id = get_dataset_identity(dataset)
        provenance["dataset_name"] = ds_id["dataset_name"]
        provenance["dataset_fingerprint"] = ds_id["dataset_fingerprint"]
        provenance["dataset_size"] = ds_id["dataset_size"]
    
    if config is not None:
        provenance["config_hash"] = compute_config_hash(config)
    
    return provenance

def is_main_process() -> bool:
    local_rank = os.environ.get("LOCAL_RANK", "0")
    rank = os.environ.get("RANK", "0")
    return local_rank == "0" and rank == "0"

def get_global_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

class RunArtifactManager:
    
    def __init__(
        self,
        run_dir: Path | str,
        run_id: Optional[str] = None,
        is_main_process_override: Optional[bool] = None,
    ):
        self.run_dir = Path(run_dir)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._is_main = is_main_process_override if is_main_process_override is not None else is_main_process()
        self._metadata: Optional[RunMetadata] = None
        
    @property
    def is_main_process(self) -> bool:
        return self._is_main
    
    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "run_metadata.json"
    
    @property
    def metadata_start_path(self) -> Path:
        return self.run_dir / "run_metadata_start.json"
    
    @property
    def metadata_final_path(self) -> Path:
        return self.run_dir / "run_metadata_final.json"
    
    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"
    
    @property
    def snapshots_dir(self) -> Path:
        return self.run_dir / "snapshots"
    
    @property
    def diffs_dir(self) -> Path:
        return self.run_dir / "diffs"
    
    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"
    
    @property
    def postmortem_path(self) -> Path:
        return self.run_dir / "postmortem.json"
    
    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.md"
    
    @property
    def reports_dir(self) -> Path:
        return self.run_dir / "reports"
    
    def initialize(
        self,
        trainer_type: str = "unknown", 
        model_name: Optional[str] = None,
        ref_model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        git_hash: Optional[str] = None,
        model: Any = None,
        ref_model: Any = None,
        tokenizer: Any = None,
        dataset: Any = None,
        save_git_diff: bool = False,
        full_package_snapshot: bool = False,
    ) -> None:
        if not self._is_main:
            return
            
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)
        self.diffs_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        provenance = collect_full_provenance(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            save_git_diff=save_git_diff,
            diff_dir=self.run_dir if save_git_diff else None,
            full_package_snapshot=full_package_snapshot,
        )
        
        final_model_name = model_name or provenance.get("model_name")
        final_model_revision = provenance.get("model_revision")
        
        final_ref_model_name = ref_model_name
        final_ref_model_revision = None
        if ref_model is not None:
            ref_id = get_model_identity(ref_model)
            final_ref_model_name = final_ref_model_name or ref_id["model_name"]
            final_ref_model_revision = ref_id["model_revision"]
        
        config_hash = None
        if config:
            config_hash = compute_config_hash(config)
        
        self._metadata = RunMetadata(
            run_id=self.run_id,
            trainer_type=trainer_type,
            model_name=final_model_name,
            model_revision=final_model_revision,
            ref_model_name=final_ref_model_name,
            ref_model_revision=final_ref_model_revision,
            tokenizer_name=provenance.get("tokenizer_name"),
            tokenizer_revision=provenance.get("tokenizer_revision"),
            dataset_name=provenance.get("dataset_name"),
            dataset_fingerprint=provenance.get("dataset_fingerprint"),
            dataset_size=provenance.get("dataset_size"),
            start_time=datetime.now(timezone.utc).isoformat(),
            status="running",
            world_size=get_world_size(),
            global_rank=get_global_rank(),
            git_sha=provenance.get("git_sha") or git_hash,
            git_branch=provenance.get("git_branch"),
            git_dirty=provenance.get("git_dirty", False),
            git_diff_path=provenance.get("git_diff_path"),
            config=config or {},
            config_hash=config_hash,
            environment=get_environment_info(),
            package_versions=provenance.get("package_versions", {}),
            hardware=provenance.get("hardware", {}),
            git_hash=git_hash,
        )
        
        self._write_metadata(self.metadata_start_path)
        self._write_metadata()
        
        header = {
            "type": "header",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "trainer_type": trainer_type,
            "config_hash": config_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
    
    def _write_metadata(self, path: Optional[Path] = None) -> None:
        if not self._is_main or self._metadata is None:
            return
        target_path = path or self.metadata_path
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata.to_dict(), f, indent=2)
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        if not self._is_main:
            return
            
        record = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    
    def save_snapshot(self, snapshot: Snapshot) -> Path:
        if not self._is_main:
            return self.snapshots_dir / f"{snapshot.metadata.step}.json"
            
        path = self.snapshots_dir / f"{snapshot.metadata.step}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        return path
    
    def load_snapshot(self, step: int) -> Optional[Snapshot]:
        path = self.snapshots_dir / f"{step}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return Snapshot.from_dict(json.load(f))
    
    def list_snapshots(self) -> List[int]:
        if not self.snapshots_dir.exists():
            return []
        steps = []
        for p in self.snapshots_dir.glob("*.json"):
            try:
                steps.append(int(p.stem))
            except ValueError:
                pass
        return sorted(steps)
    
    def save_diff(self, diff: SnapshotDiff) -> Path:
        if not self._is_main:
            return self.diffs_dir / f"{diff.step_a}_to_{diff.step_b}.json"
            
        path = self.diffs_dir / f"{diff.step_a}_to_{diff.step_b}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(diff.to_dict(), f, indent=2)
        return path
    
    def save_postmortem(self, postmortem: Postmortem) -> Path:
        if not self._is_main:
            return self.postmortem_path
            
        with open(self.postmortem_path, "w", encoding="utf-8") as f:
            json.dump(postmortem.to_dict(), f, indent=2)
        return self.postmortem_path
    
    def finalize(self, status: str = "completed", total_steps: int = 0) -> None:
        if not self._is_main:
            return
            
        if self._metadata:
            self._metadata.status = status
            self._metadata.total_steps = total_steps
            self._metadata.end_time = datetime.now(timezone.utc).isoformat()
            self._write_metadata()
            self._write_metadata(self.metadata_final_path)
        
        footer = {
            "type": "footer",
            "run_id": self.run_id,
            "status": status,
            "total_steps": total_steps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(footer) + "\n")
