"""Run artifact schema and directory management.

Defines the canonical structure for post-training run artifacts:
- run_metadata.json: Run configuration and environment info
- metrics.jsonl: Step-level training metrics
- snapshots/{step}.json: Behavior snapshots at intervals
- diffs/{a}_to_{b}.json: Behavior diffs between snapshots
- postmortem.json: Crash/termination info (if applicable)
- report.md: Final diagnostic report

Only global rank 0 should write artifacts in distributed settings.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Schema version for forward compatibility
ARTIFACT_SCHEMA_VERSION = "1.0"


@dataclass
class RunMetadata:
    """Metadata for a training run with comprehensive provenance.
    
    Provenance fields capture everything needed to reproduce and audit a run:
    - Git state (SHA, dirty flag, optional diff)
    - Full resolved config with stable hash
    - Package versions (core + full environment)
    - Hardware/runtime info (GPU, CUDA, torch, hostname, world size)
    - Model/tokenizer identity (path or Hub ID + revision)
    - Dataset identity (fingerprint/hash or user-provided ID)
    """
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    run_id: str = ""
    trainer_type: str = "unknown"
    
    # Model identity
    model_name: Optional[str] = None
    model_revision: Optional[str] = None  # Hub commit/revision if available
    ref_model_name: Optional[str] = None
    ref_model_revision: Optional[str] = None
    tokenizer_name: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    
    # Dataset identity
    dataset_name: Optional[str] = None
    dataset_fingerprint: Optional[str] = None  # HF datasets fingerprint if available
    dataset_size: Optional[int] = None  # Number of examples
    
    # Timing
    start_time: str = ""
    end_time: Optional[str] = None
    status: str = "running"  # running, completed, crashed, interrupted
    total_steps: int = 0
    
    # Distributed training
    world_size: int = 1
    global_rank: int = 0
    
    # Git provenance
    git_sha: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    git_diff_path: Optional[str] = None  # Path to saved diff file (if dirty)
    
    # Config with hash for reproducibility
    config: Dict[str, Any] = field(default_factory=dict)
    config_hash: Optional[str] = None  # Stable hash of resolved config
    
    # Environment and hardware
    environment: Dict[str, Any] = field(default_factory=dict)
    package_versions: Dict[str, str] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    
    # Legacy field for backwards compatibility
    git_hash: Optional[str] = None  # Deprecated: use git_sha
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunMetadata":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass 
class SnapshotMetadata:
    """Metadata for a behavior snapshot."""
    step: int
    timestamp: str
    num_prompts: int
    generation_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SnapshotEntry:
    """Single prompt result in a snapshot."""
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
    """Complete behavior snapshot."""
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
    """Per-prompt diff between two snapshots."""
    prompt_id: str
    length_delta: int
    length_pct_change: float
    refusal_changed: bool  # True if refusal status flipped
    refusal_before: bool
    refusal_after: bool
    entropy_delta: Optional[float] = None
    logprob_delta: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SnapshotDiff:
    """Diff between two snapshots."""
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
    """Postmortem record for crashed/interrupted runs."""
    exit_reason: str  # oom, sigterm, sigint, nan, divergence, exception, unknown
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
    """Collect basic environment metadata (legacy, use get_full_provenance for comprehensive)."""
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
    }
    
    # Try to get GPU info
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
    
    # Try to get transformers/trl versions
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
    """Collect git repository information.
    
    Args:
        repo_path: Path to git repository (defaults to current directory)
        save_diff: If True and repo is dirty, save diff to file
        diff_dir: Directory to save diff file (required if save_diff=True)
        
    Returns:
        Dict with git_sha, git_branch, git_dirty, git_diff_path
    """
    import subprocess
    
    git_info: Dict[str, Any] = {
        "git_sha": None,
        "git_branch": None,
        "git_dirty": False,
        "git_diff_path": None,
    }
    
    try:
        cwd = str(repo_path) if repo_path else None
        
        # Get current SHA
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=cwd, timeout=5
        )
        if result.returncode == 0:
            git_info["git_sha"] = result.stdout.strip()
        
        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=cwd, timeout=5
        )
        if result.returncode == 0:
            git_info["git_branch"] = result.stdout.strip()
        
        # Check if dirty (uncommitted changes)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=cwd, timeout=5
        )
        if result.returncode == 0:
            git_info["git_dirty"] = len(result.stdout.strip()) > 0
            
            # Save diff if requested and dirty
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
        pass  # Git not available or not a repo
    
    return git_info


def get_hardware_info() -> Dict[str, Any]:
    """Collect comprehensive hardware and runtime information."""
    import subprocess
    
    hw_info: Dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version,
    }
    
    # GPU information
    try:
        import torch
        hw_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw_info["cuda_version"] = torch.version.cuda
            hw_info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
            hw_info["gpu_count"] = torch.cuda.device_count()
            hw_info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            # GPU memory
            hw_info["gpu_memory_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        hw_info["cuda_available"] = False
    
    # Distributed training info
    hw_info["world_size"] = get_world_size()
    hw_info["global_rank"] = get_global_rank()
    hw_info["local_rank"] = int(os.environ.get("LOCAL_RANK", "0"))
    
    return hw_info


def get_package_versions(full_env: bool = False) -> Dict[str, str]:
    """Collect package version information.
    
    Args:
        full_env: If True, collect all installed packages (slower)
        
    Returns:
        Dict mapping package name to version string
    """
    versions: Dict[str, str] = {}
    
    # Core packages (always collected)
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
    
    # Full environment snapshot
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
    """Compute a stable hash of a configuration dict.
    
    The hash is stable across runs for the same logical config,
    allowing detection of config changes on resume.
    """
    import hashlib
    
    def _normalize(obj: Any) -> Any:
        """Normalize object for stable hashing."""
        if isinstance(obj, dict):
            return {k: _normalize(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, (list, tuple)):
            return [_normalize(x) for x in obj]
        elif isinstance(obj, float):
            return round(obj, 10)  # Avoid float precision issues
        elif hasattr(obj, "__dict__"):
            return _normalize(obj.__dict__)
        else:
            return obj
    
    normalized = _normalize(config)
    config_str = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def get_model_identity(model: Any) -> Dict[str, Any]:
    """Extract model identity information.
    
    Args:
        model: HuggingFace model or path string
        
    Returns:
        Dict with model_name, model_revision
    """
    identity: Dict[str, Any] = {
        "model_name": None,
        "model_revision": None,
    }
    
    if isinstance(model, str):
        identity["model_name"] = model
        return identity
    
    # Try to get name/path from model
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
    """Extract tokenizer identity information.
    
    Args:
        tokenizer: HuggingFace tokenizer or path string
        
    Returns:
        Dict with tokenizer_name, tokenizer_revision
    """
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
    """Extract dataset identity information.
    
    Args:
        dataset: HuggingFace dataset or any dataset object
        
    Returns:
        Dict with dataset_name, dataset_fingerprint, dataset_size
    """
    identity: Dict[str, Any] = {
        "dataset_name": None,
        "dataset_fingerprint": None,
        "dataset_size": None,
    }
    
    if dataset is None:
        return identity
    
    # HuggingFace datasets
    if hasattr(dataset, "_fingerprint"):
        identity["dataset_fingerprint"] = dataset._fingerprint
    
    if hasattr(dataset, "info") and dataset.info:
        info = dataset.info
        if hasattr(info, "dataset_name"):
            identity["dataset_name"] = info.dataset_name
    
    # Try to get size
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
    """Collect comprehensive provenance information for a training run.
    
    This is the main entry point for provenance collection, gathering:
    - Git state (SHA, branch, dirty flag, optional diff)
    - Hardware/runtime info (GPU, CUDA, torch, cudnn, hostname)
    - Package versions (core + optional full environment)
    - Config hash for reproducibility
    - Model/tokenizer/dataset identity
    
    Args:
        model: Model being trained (for identity extraction)
        tokenizer: Tokenizer being used
        dataset: Training dataset
        config: Training configuration dict
        repo_path: Git repository path (defaults to current directory)
        save_git_diff: Save uncommitted changes to file
        diff_dir: Directory for saving git diff
        full_package_snapshot: Include all installed packages (slower)
        
    Returns:
        Dict with all provenance fields
    """
    provenance: Dict[str, Any] = {}
    
    # Git info
    git_info = get_git_info(repo_path, save_git_diff, diff_dir)
    provenance.update(git_info)
    
    # Hardware info
    provenance["hardware"] = get_hardware_info()
    
    # Package versions
    provenance["package_versions"] = get_package_versions(full_package_snapshot)
    
    # Model identity
    if model is not None:
        model_id = get_model_identity(model)
        provenance["model_name"] = model_id["model_name"]
        provenance["model_revision"] = model_id["model_revision"]
    
    # Tokenizer identity
    if tokenizer is not None:
        tok_id = get_tokenizer_identity(tokenizer)
        provenance["tokenizer_name"] = tok_id["tokenizer_name"]
        provenance["tokenizer_revision"] = tok_id["tokenizer_revision"]
    
    # Dataset identity
    if dataset is not None:
        ds_id = get_dataset_identity(dataset)
        provenance["dataset_name"] = ds_id["dataset_name"]
        provenance["dataset_fingerprint"] = ds_id["dataset_fingerprint"]
        provenance["dataset_size"] = ds_id["dataset_size"]
    
    # Config hash
    if config is not None:
        provenance["config_hash"] = compute_config_hash(config)
    
    return provenance


def is_main_process() -> bool:
    """Best-effort check for main process (rank 0) in distributed training.
    
    Returns True if:
    - Not in distributed mode
    - LOCAL_RANK or RANK env var is 0 or unset
    """
    local_rank = os.environ.get("LOCAL_RANK", "0")
    rank = os.environ.get("RANK", "0")
    return local_rank == "0" and rank == "0"


def get_global_rank() -> int:
    """Get global rank in distributed training."""
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    """Get world size in distributed training."""
    return int(os.environ.get("WORLD_SIZE", "1"))


class RunArtifactManager:
    """Manages the run artifact directory structure.
    
    Creates immutable provenance artifacts:
    - run_metadata_start.json: Written at training start (immutable)
    - run_metadata.json: Updated throughout training (current state)
    - run_metadata_final.json: Written at training end (immutable)
    
    Only writes artifacts if this is the main process (rank 0).
    """
    
    def __init__(
        self,
        run_dir: Path | str,
        run_id: Optional[str] = None,
        is_main_process_override: Optional[bool] = None,
    ):
        """Initialize artifact manager.
        
        Args:
            run_dir: Root directory for run artifacts
            run_id: Optional run identifier (auto-generated if not provided)
            is_main_process_override: Explicit flag for main process status (e.g., from TrainerState)
        """
        self.run_dir = Path(run_dir)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._is_main = is_main_process_override if is_main_process_override is not None else is_main_process()
        self._metadata: Optional[RunMetadata] = None
        
    @property
    def is_main_process(self) -> bool:
        """Whether this process should write artifacts."""
        return self._is_main
    
    @property
    def metadata_path(self) -> Path:
        """Current run metadata (updated during training)."""
        return self.run_dir / "run_metadata.json"
    
    @property
    def metadata_start_path(self) -> Path:
        """Immutable start metadata (written once at start)."""
        return self.run_dir / "run_metadata_start.json"
    
    @property
    def metadata_final_path(self) -> Path:
        """Immutable final metadata (written once at end)."""
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
        # New provenance parameters
        model: Any = None,
        ref_model: Any = None,
        tokenizer: Any = None,
        dataset: Any = None,
        save_git_diff: bool = False,
        full_package_snapshot: bool = False,
    ) -> None:
        """Initialize the run directory and write immutable start metadata.
        
        This creates the run directory structure and writes comprehensive
        provenance information to run_metadata_start.json (immutable).
        
        Args:
            trainer_type: Type of trainer (dpo, ppo, sft, etc.)
            model_name: Name/path of model being trained (legacy, prefer model=)
            ref_model_name: Name/path of reference model (legacy, prefer ref_model=)
            config: Training configuration dict
            git_hash: Git commit hash (legacy, auto-detected if not provided)
            model: Model object for identity extraction
            ref_model: Reference model object for identity extraction
            tokenizer: Tokenizer object for identity extraction
            dataset: Dataset object for identity extraction
            save_git_diff: Save uncommitted git changes to file
            full_package_snapshot: Include all installed packages in provenance
        """
        if not self._is_main:
            return
            
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)
        self.diffs_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Collect full provenance
        provenance = collect_full_provenance(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            save_git_diff=save_git_diff,
            diff_dir=self.run_dir if save_git_diff else None,
            full_package_snapshot=full_package_snapshot,
        )
        
        # Extract model identity - use explicit params or extracted from objects
        final_model_name = model_name or provenance.get("model_name")
        final_model_revision = provenance.get("model_revision")
        
        # Extract ref model identity
        final_ref_model_name = ref_model_name
        final_ref_model_revision = None
        if ref_model is not None:
            ref_id = get_model_identity(ref_model)
            final_ref_model_name = final_ref_model_name or ref_id["model_name"]
            final_ref_model_revision = ref_id["model_revision"]
        
        # Compute config hash
        config_hash = None
        if config:
            config_hash = compute_config_hash(config)
        
        # Create metadata with full provenance
        self._metadata = RunMetadata(
            run_id=self.run_id,
            trainer_type=trainer_type,
            # Model identity
            model_name=final_model_name,
            model_revision=final_model_revision,
            ref_model_name=final_ref_model_name,
            ref_model_revision=final_ref_model_revision,
            tokenizer_name=provenance.get("tokenizer_name"),
            tokenizer_revision=provenance.get("tokenizer_revision"),
            # Dataset identity
            dataset_name=provenance.get("dataset_name"),
            dataset_fingerprint=provenance.get("dataset_fingerprint"),
            dataset_size=provenance.get("dataset_size"),
            # Timing
            start_time=datetime.now(timezone.utc).isoformat(),
            status="running",
            # Distributed
            world_size=get_world_size(),
            global_rank=get_global_rank(),
            # Git provenance
            git_sha=provenance.get("git_sha") or git_hash,
            git_branch=provenance.get("git_branch"),
            git_dirty=provenance.get("git_dirty", False),
            git_diff_path=provenance.get("git_diff_path"),
            # Config
            config=config or {},
            config_hash=config_hash,
            # Environment
            environment=get_environment_info(),  # Legacy field
            package_versions=provenance.get("package_versions", {}),
            hardware=provenance.get("hardware", {}),
            # Legacy
            git_hash=git_hash,
        )
        
        # Write immutable start metadata
        self._write_metadata(self.metadata_start_path)
        # Also write current metadata (will be updated)
        self._write_metadata()
        
        # Initialize metrics file with header
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
        """Write metadata to file.
        
        Args:
            path: Override path (defaults to metadata_path)
        """
        if not self._is_main or self._metadata is None:
            return
        target_path = path or self.metadata_path
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata.to_dict(), f, indent=2)
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Append metrics for a step to the metrics log.
        
        Args:
            step: Training step number
            metrics: Dict of metric name -> value
        """
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
        """Save a behavior snapshot.
        
        Args:
            snapshot: Snapshot object to save
            
        Returns:
            Path to saved snapshot file
        """
        if not self._is_main:
            return self.snapshots_dir / f"{snapshot.metadata.step}.json"
            
        path = self.snapshots_dir / f"{snapshot.metadata.step}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        return path
    
    def load_snapshot(self, step: int) -> Optional[Snapshot]:
        """Load a snapshot for a given step.
        
        Args:
            step: Step number to load
            
        Returns:
            Snapshot object or None if not found
        """
        path = self.snapshots_dir / f"{step}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return Snapshot.from_dict(json.load(f))
    
    def list_snapshots(self) -> List[int]:
        """List all available snapshot steps.
        
        Returns:
            Sorted list of step numbers with snapshots
        """
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
        """Save a snapshot diff.
        
        Args:
            diff: SnapshotDiff object to save
            
        Returns:
            Path to saved diff file
        """
        if not self._is_main:
            return self.diffs_dir / f"{diff.step_a}_to_{diff.step_b}.json"
            
        path = self.diffs_dir / f"{diff.step_a}_to_{diff.step_b}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(diff.to_dict(), f, indent=2)
        return path
    
    def save_postmortem(self, postmortem: Postmortem) -> Path:
        """Save postmortem data.
        
        Args:
            postmortem: Postmortem object to save
            
        Returns:
            Path to saved postmortem file
        """
        if not self._is_main:
            return self.postmortem_path
            
        with open(self.postmortem_path, "w", encoding="utf-8") as f:
            json.dump(postmortem.to_dict(), f, indent=2)
        return self.postmortem_path
    
    def finalize(self, status: str = "completed", total_steps: int = 0) -> None:
        """Finalize the run artifacts and write immutable final metadata.
        
        Args:
            status: Final run status (completed, crashed, interrupted)
            total_steps: Total training steps completed
        """
        if not self._is_main:
            return
            
        # Update metadata
        if self._metadata:
            self._metadata.status = status
            self._metadata.total_steps = total_steps
            self._metadata.end_time = datetime.now(timezone.utc).isoformat()
            # Write final current metadata
            self._write_metadata()
            # Write immutable final metadata
            self._write_metadata(self.metadata_final_path)
        
        # Write footer to metrics
        footer = {
            "type": "footer",
            "run_id": self.run_id,
            "status": status,
            "total_steps": total_steps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(footer) + "\n")
