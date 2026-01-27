"""GPU profiling for training performance.

Tracks:
- GPU memory usage and growth
- GPU utilization percentage
- Memory fragmentation indicators
- Multi-GPU load balance

Works with CUDA when available, gracefully degrades otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager


@dataclass
class GPUMemorySnapshot:
    """Snapshot of GPU memory state."""
    step: int
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    device_id: int = 0
    
    @property
    def fragmentation_ratio(self) -> float:
        """Estimate memory fragmentation (reserved but not allocated)."""
        if self.reserved_mb == 0:
            return 0.0
        return 1.0 - (self.allocated_mb / self.reserved_mb)


@dataclass
class GPUUtilizationSample:
    """GPU utilization sample."""
    timestamp: float
    gpu_util_percent: float
    memory_util_percent: float
    device_id: int = 0


@dataclass
class GPUProfileReport:
    """Comprehensive GPU profiling report."""
    # Memory
    peak_memory_mb: float
    final_memory_mb: float
    memory_growth_mb: float
    avg_fragmentation: float
    
    # Utilization (may be None if not available)
    avg_gpu_util: Optional[float]
    avg_memory_util: Optional[float]
    
    # Recommendations
    memory_pressure: str  # "low", "moderate", "high"
    recommendations: List[str]
    
    def format(self) -> str:
        """Format as human-readable string."""
        lines = [
            "GPU Profile Report",
            "=" * 40,
            f"Peak memory: {self.peak_memory_mb:,.0f} MB",
            f"Final memory: {self.final_memory_mb:,.0f} MB",
            f"Memory growth: {self.memory_growth_mb:+,.0f} MB",
            f"Avg fragmentation: {self.avg_fragmentation:.1%}",
        ]
        
        if self.avg_gpu_util is not None:
            lines.append(f"Avg GPU utilization: {self.avg_gpu_util:.0f}%")
        if self.avg_memory_util is not None:
            lines.append(f"Avg memory utilization: {self.avg_memory_util:.0f}%")
            
        lines.append(f"Memory pressure: {self.memory_pressure.upper()}")
        
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
                
        return "\n".join(lines)


def _get_torch_cuda():
    """Safely import torch.cuda."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda
    except ImportError:
        pass
    return None


def _get_pynvml():
    """Safely import pynvml for GPU utilization."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except (ImportError, Exception):
        return None


class GPUProfiler:
    """Profiles GPU memory and utilization during training.
    
    Automatically detects available GPU monitoring capabilities:
    - Uses torch.cuda for memory tracking (if available)
    - Uses pynvml for utilization tracking (if available)
    - Gracefully degrades if neither available
    
    Usage:
        profiler = GPUProfiler()
        
        for step in range(num_steps):
            # ... training step ...
            profiler.record_step(step)
            
        print(profiler.report().format())
    """
    
    def __init__(self, device_id: int = 0, sample_utilization: bool = True):
        """Initialize GPU profiler.
        
        Args:
            device_id: CUDA device ID to profile
            sample_utilization: Whether to sample GPU utilization (requires pynvml)
        """
        self.device_id = device_id
        self._torch_cuda = _get_torch_cuda()
        self._pynvml = _get_pynvml() if sample_utilization else None
        
        self._memory_snapshots: List[GPUMemorySnapshot] = []
        self._utilization_samples: List[GPUUtilizationSample] = []
        
        self._available = self._torch_cuda is not None
        
    @property
    def available(self) -> bool:
        """Whether GPU profiling is available."""
        return self._available
        
    def record_step(self, step: int) -> Optional[GPUMemorySnapshot]:
        """Record GPU state at a training step.
        
        Args:
            step: Current training step
            
        Returns:
            GPUMemorySnapshot if GPU available, None otherwise
        """
        if not self._available:
            return None
            
        try:
            cuda = self._torch_cuda
            snapshot = GPUMemorySnapshot(
                step=step,
                timestamp=time.time(),
                allocated_mb=cuda.memory_allocated(self.device_id) / 1024 / 1024,
                reserved_mb=cuda.memory_reserved(self.device_id) / 1024 / 1024,
                max_allocated_mb=cuda.max_memory_allocated(self.device_id) / 1024 / 1024,
                device_id=self.device_id,
            )
            self._memory_snapshots.append(snapshot)
            
            # Also sample utilization if available
            self._sample_utilization()
            
            return snapshot
            
        except Exception:
            return None
            
    def _sample_utilization(self) -> None:
        """Sample GPU utilization using pynvml."""
        if self._pynvml is None:
            return
            
        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            sample = GPUUtilizationSample(
                timestamp=time.time(),
                gpu_util_percent=util.gpu,
                memory_util_percent=util.memory,
                device_id=self.device_id,
            )
            self._utilization_samples.append(sample)
            
        except Exception:
            pass
            
    def get_current_memory_mb(self) -> Optional[float]:
        """Get current GPU memory allocated in MB."""
        if not self._available:
            return None
        try:
            return self._torch_cuda.memory_allocated(self.device_id) / 1024 / 1024
        except Exception:
            return None
            
    def get_peak_memory_mb(self) -> Optional[float]:
        """Get peak GPU memory allocated in MB."""
        if not self._available:
            return None
        try:
            return self._torch_cuda.max_memory_allocated(self.device_id) / 1024 / 1024
        except Exception:
            return None
            
    def get_memory_growth(self) -> Optional[float]:
        """Calculate memory growth from start to current.
        
        Returns:
            Memory growth in MB, or None if not enough data
        """
        if len(self._memory_snapshots) < 2:
            return None
            
        start = self._memory_snapshots[0].allocated_mb
        end = self._memory_snapshots[-1].allocated_mb
        return end - start
        
    def detect_memory_leak(self, threshold_mb_per_step: float = 1.0) -> bool:
        """Detect potential memory leak.
        
        Args:
            threshold_mb_per_step: MB growth per step to consider a leak
            
        Returns:
            True if memory appears to be leaking
        """
        if len(self._memory_snapshots) < 10:
            return False
            
        # Look at memory trend over last half of training
        mid = len(self._memory_snapshots) // 2
        first_half = [s.allocated_mb for s in self._memory_snapshots[:mid]]
        second_half = [s.allocated_mb for s in self._memory_snapshots[mid:]]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        growth_per_step = (avg_second - avg_first) / mid
        return growth_per_step > threshold_mb_per_step
        
    def report(self) -> GPUProfileReport:
        """Generate comprehensive GPU profile report."""
        if not self._memory_snapshots:
            return GPUProfileReport(
                peak_memory_mb=0,
                final_memory_mb=0,
                memory_growth_mb=0,
                avg_fragmentation=0,
                avg_gpu_util=None,
                avg_memory_util=None,
                memory_pressure="unknown",
                recommendations=["No GPU data collected"],
            )
            
        # Memory stats
        peak = max(s.max_allocated_mb for s in self._memory_snapshots)
        final = self._memory_snapshots[-1].allocated_mb
        growth = self.get_memory_growth() or 0
        avg_frag = sum(s.fragmentation_ratio for s in self._memory_snapshots) / len(self._memory_snapshots)
        
        # Utilization stats
        avg_gpu_util = None
        avg_mem_util = None
        if self._utilization_samples:
            avg_gpu_util = sum(s.gpu_util_percent for s in self._utilization_samples) / len(self._utilization_samples)
            avg_mem_util = sum(s.memory_util_percent for s in self._utilization_samples) / len(self._utilization_samples)
            
        # Determine memory pressure
        if self._available:
            try:
                total_mem = self._torch_cuda.get_device_properties(self.device_id).total_memory / 1024 / 1024
                usage_ratio = peak / total_mem
                if usage_ratio > 0.9:
                    memory_pressure = "high"
                elif usage_ratio > 0.7:
                    memory_pressure = "moderate"
                else:
                    memory_pressure = "low"
            except Exception:
                memory_pressure = "unknown"
        else:
            memory_pressure = "unknown"
            
        # Generate recommendations
        recommendations = []
        
        if memory_pressure == "high":
            recommendations.append("Enable gradient_checkpointing to reduce memory usage")
            recommendations.append("Consider reducing batch size")
            
        if avg_frag > 0.3:
            recommendations.append(f"High memory fragmentation ({avg_frag:.0%}). Try torch.cuda.empty_cache() or restart process.")
            
        if growth > 500:  # >500MB growth
            recommendations.append(f"Memory grew {growth:.0f}MB during training. Check for memory leaks.")
            
        if self.detect_memory_leak():
            recommendations.append("⚠️ Potential memory leak detected!")
            
        if avg_gpu_util is not None and avg_gpu_util < 50:
            recommendations.append(f"Low GPU utilization ({avg_gpu_util:.0f}%). May be I/O or CPU bound.")
            
        if not recommendations:
            recommendations.append("GPU utilization looks healthy.")
            
        return GPUProfileReport(
            peak_memory_mb=peak,
            final_memory_mb=final,
            memory_growth_mb=growth,
            avg_fragmentation=avg_frag,
            avg_gpu_util=avg_gpu_util,
            avg_memory_util=avg_mem_util,
            memory_pressure=memory_pressure,
            recommendations=recommendations,
        )
        
    @contextmanager
    def track_operation(self, name: str):
        """Track memory for a specific operation.
        
        Useful for identifying which operations use most memory.
        
        Usage:
            with profiler.track_operation("forward_pass"):
                outputs = model(inputs)
        """
        if not self._available:
            yield
            return
            
        try:
            self._torch_cuda.reset_peak_memory_stats(self.device_id)
            before = self._torch_cuda.memory_allocated(self.device_id)
            
            yield
            
            after = self._torch_cuda.memory_allocated(self.device_id)
            peak = self._torch_cuda.max_memory_allocated(self.device_id)
            
            # Could store these for later analysis
            # For now, just tracking
            
        except Exception:
            yield


def get_gpu_summary() -> Dict[str, any]:
    """Get quick summary of GPU state.
    
    Returns:
        Dict with GPU info, or empty dict if no GPU
    """
    cuda = _get_torch_cuda()
    if cuda is None:
        return {}
        
    try:
        return {
            "device_count": cuda.device_count(),
            "current_device": cuda.current_device(),
            "device_name": cuda.get_device_name(0),
            "memory_allocated_mb": cuda.memory_allocated(0) / 1024 / 1024,
            "memory_reserved_mb": cuda.memory_reserved(0) / 1024 / 1024,
            "max_memory_mb": cuda.get_device_properties(0).total_memory / 1024 / 1024,
        }
    except Exception:
        return {}


# =============================================================================
# Multi-GPU Monitoring
# =============================================================================


@dataclass
class GPUDeviceStatus:
    """Status of a single GPU device."""
    device_id: int
    name: str
    gpu_util_percent: float
    memory_util_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: Optional[float] = None
    power_watts: Optional[float] = None
    
    @property
    def memory_free_mb(self) -> float:
        """Free memory in MB."""
        return self.memory_total_mb - self.memory_used_mb
    
    @property
    def is_idle(self) -> bool:
        """Whether GPU appears idle (< 5% utilization)."""
        return self.gpu_util_percent < 5.0
    
    @property
    def is_active(self) -> bool:
        """Whether GPU is actively computing (> 50% utilization)."""
        return self.gpu_util_percent >= 50.0


@dataclass 
class MultiGPUSnapshot:
    """Snapshot of all GPUs on the system."""
    timestamp: float
    devices: List[GPUDeviceStatus]
    
    @property
    def device_count(self) -> int:
        """Number of GPUs."""
        return len(self.devices)
    
    @property
    def active_count(self) -> int:
        """Number of GPUs with >50% utilization."""
        return sum(1 for d in self.devices if d.is_active)
    
    @property
    def idle_count(self) -> int:
        """Number of GPUs with <5% utilization."""
        return sum(1 for d in self.devices if d.is_idle)
    
    @property
    def avg_utilization(self) -> float:
        """Average GPU utilization across all devices."""
        if not self.devices:
            return 0.0
        return sum(d.gpu_util_percent for d in self.devices) / len(self.devices)
    
    @property
    def min_utilization(self) -> float:
        """Minimum GPU utilization."""
        if not self.devices:
            return 0.0
        return min(d.gpu_util_percent for d in self.devices)
    
    @property
    def max_utilization(self) -> float:
        """Maximum GPU utilization."""
        if not self.devices:
            return 0.0
        return max(d.gpu_util_percent for d in self.devices)
    
    def get_idle_devices(self) -> List[GPUDeviceStatus]:
        """Get list of idle GPUs."""
        return [d for d in self.devices if d.is_idle]
    
    def get_active_devices(self) -> List[GPUDeviceStatus]:
        """Get list of active GPUs."""
        return [d for d in self.devices if d.is_active]


@dataclass
class GPUImbalanceReport:
    """Report on GPU utilization imbalance."""
    has_imbalance: bool
    idle_gpus: List[int]  # Device IDs of idle GPUs
    active_gpus: List[int]  # Device IDs of active GPUs
    utilization_spread: float  # max - min utilization
    avg_utilization: float
    severity: str  # "none", "minor", "moderate", "severe"
    message: str
    
    def format(self) -> str:
        """Format as human-readable string."""
        lines = [
            "GPU Imbalance Report",
            "=" * 50,
            f"Active GPUs: {len(self.active_gpus)} / {len(self.active_gpus) + len(self.idle_gpus)}",
            f"Avg utilization: {self.avg_utilization:.1f}%",
            f"Utilization spread: {self.utilization_spread:.1f}%",
            f"Severity: {self.severity.upper()}",
        ]
        
        if self.idle_gpus:
            lines.append(f"⚠️  Idle GPUs: {self.idle_gpus}")
        
        lines.append(f"\n{self.message}")
        
        return "\n".join(lines)


class MultiGPUMonitor:
    """Monitor utilization across all GPUs on the system.
    
    Uses pynvml to query GPU state. Can detect:
    - Which GPUs are idle vs active
    - Utilization imbalance (e.g., 7 GPUs working, 1 stuck)
    - Memory usage across all devices
    
    Example:
        monitor = MultiGPUMonitor()
        
        # Get current state of all GPUs
        snapshot = monitor.snapshot()
        print(f"Active: {snapshot.active_count}/{snapshot.device_count}")
        
        # Check for imbalance
        report = monitor.check_imbalance()
        if report.has_imbalance:
            print(f"⚠️ GPU imbalance: {report.idle_gpus} are idle!")
        
        # Pretty print all GPU status
        print(monitor.format_status())
    """
    
    def __init__(self, imbalance_threshold: float = 50.0):
        """Initialize multi-GPU monitor.
        
        Args:
            imbalance_threshold: Utilization difference (%) to flag as imbalance.
                                 E.g., 50.0 means if one GPU is at 90% and another
                                 at 30%, that's a 60% spread and would be flagged.
        """
        self.imbalance_threshold = imbalance_threshold
        self._pynvml = _get_pynvml()
        self._available = self._pynvml is not None
        self._device_count = 0
        
        if self._available:
            try:
                self._device_count = self._pynvml.nvmlDeviceGetCount()
            except Exception:
                self._available = False
    
    @property
    def available(self) -> bool:
        """Whether GPU monitoring is available."""
        return self._available
    
    @property
    def device_count(self) -> int:
        """Number of GPUs detected."""
        return self._device_count
    
    def snapshot(self) -> Optional[MultiGPUSnapshot]:
        """Take a snapshot of all GPU states.
        
        Returns:
            MultiGPUSnapshot with status of all GPUs, or None if unavailable.
        """
        if not self._available:
            return None
        
        devices = []
        timestamp = time.time()
        
        for i in range(self._device_count):
            try:
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic info
                name = self._pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Utilization
                util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Memory
                mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Temperature (optional)
                try:
                    temp = self._pynvml.nvmlDeviceGetTemperature(
                        handle, self._pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp = None
                
                # Power (optional)
                try:
                    power = self._pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except Exception:
                    power = None
                
                devices.append(GPUDeviceStatus(
                    device_id=i,
                    name=name,
                    gpu_util_percent=util.gpu,
                    memory_util_percent=util.memory,
                    memory_used_mb=mem_info.used / 1024 / 1024,
                    memory_total_mb=mem_info.total / 1024 / 1024,
                    temperature_c=temp,
                    power_watts=power,
                ))
                
            except Exception as e:
                # If one GPU fails, add placeholder
                devices.append(GPUDeviceStatus(
                    device_id=i,
                    name=f"GPU {i} (error)",
                    gpu_util_percent=0.0,
                    memory_util_percent=0.0,
                    memory_used_mb=0.0,
                    memory_total_mb=0.0,
                ))
        
        return MultiGPUSnapshot(timestamp=timestamp, devices=devices)
    
    def check_imbalance(self) -> Optional[GPUImbalanceReport]:
        """Check for GPU utilization imbalance.
        
        Detects scenarios like:
        - One GPU stuck/idle while others are working
        - Uneven load distribution
        
        Returns:
            GPUImbalanceReport, or None if monitoring unavailable.
        """
        snapshot = self.snapshot()
        if snapshot is None or snapshot.device_count == 0:
            return None
        
        idle_gpus = [d.device_id for d in snapshot.get_idle_devices()]
        active_gpus = [d.device_id for d in snapshot.get_active_devices()]
        
        spread = snapshot.max_utilization - snapshot.min_utilization
        avg_util = snapshot.avg_utilization
        
        # Determine severity
        has_imbalance = False
        severity = "none"
        message = "All GPUs are balanced."
        
        if snapshot.device_count > 1:
            # Check if some GPUs are idle while others are active
            if idle_gpus and active_gpus:
                has_imbalance = True
                severity = "severe"
                message = (
                    f"⚠️ {len(idle_gpus)} GPU(s) idle while {len(active_gpus)} are active! "
                    f"Idle: {idle_gpus}. This may indicate a stuck process, "
                    f"NCCL hang, or load imbalance."
                )
            elif spread >= self.imbalance_threshold:
                has_imbalance = True
                if spread >= 70:
                    severity = "severe"
                elif spread >= 50:
                    severity = "moderate"
                else:
                    severity = "minor"
                
                message = (
                    f"Utilization spread of {spread:.0f}% across GPUs. "
                    f"Min: {snapshot.min_utilization:.0f}%, Max: {snapshot.max_utilization:.0f}%."
                )
        
        return GPUImbalanceReport(
            has_imbalance=has_imbalance,
            idle_gpus=idle_gpus,
            active_gpus=active_gpus,
            utilization_spread=spread,
            avg_utilization=avg_util,
            severity=severity,
            message=message,
        )
    
    def format_status(self, compact: bool = False) -> str:
        """Format current GPU status as a string.
        
        Args:
            compact: If True, single-line format. If False, detailed multi-line.
            
        Returns:
            Formatted string showing all GPU states.
        """
        snapshot = self.snapshot()
        if snapshot is None:
            return "GPU monitoring unavailable (pynvml not installed or no GPUs)"
        
        if compact:
            # Single line: GPU 0: 94% | GPU 1: 93% | GPU 2: 0% ⚠️ | ...
            parts = []
            for d in snapshot.devices:
                status = f"GPU {d.device_id}: {d.gpu_util_percent:.0f}%"
                if d.is_idle and snapshot.active_count > 0:
                    status += " ⚠️"
                parts.append(status)
            return " | ".join(parts)
        
        else:
            # Detailed multi-line format
            lines = [
                f"GPU Status ({snapshot.device_count} devices)",
                "=" * 60,
            ]
            
            for d in snapshot.devices:
                idle_marker = " ⚠️ IDLE" if d.is_idle and snapshot.active_count > 0 else ""
                lines.append(
                    f"  GPU {d.device_id} ({d.name}){idle_marker}"
                )
                lines.append(
                    f"    Compute: {d.gpu_util_percent:5.1f}%  |  "
                    f"Memory: {d.memory_used_mb:,.0f} / {d.memory_total_mb:,.0f} MB ({d.memory_util_percent:.0f}%)"
                )
                
                extras = []
                if d.temperature_c is not None:
                    extras.append(f"Temp: {d.temperature_c}°C")
                if d.power_watts is not None:
                    extras.append(f"Power: {d.power_watts:.0f}W")
                if extras:
                    lines.append(f"    {' | '.join(extras)}")
                lines.append("")
            
            # Summary
            lines.append(f"Summary: {snapshot.active_count} active, {snapshot.idle_count} idle")
            lines.append(f"Avg utilization: {snapshot.avg_utilization:.1f}%")
            
            return "\n".join(lines)


def get_all_gpu_utilization() -> Optional[List[Dict[str, float]]]:
    """Quick function to get utilization of all GPUs.
    
    Returns:
        List of dicts with gpu_util and memory_util per device,
        or None if unavailable.
        
    Example:
        >>> get_all_gpu_utilization()
        [
            {'device_id': 0, 'gpu_util': 94.0, 'memory_util': 45.0},
            {'device_id': 1, 'gpu_util': 92.0, 'memory_util': 44.0},
            {'device_id': 2, 'gpu_util': 0.0, 'memory_util': 5.0},  # ← stuck!
            ...
        ]
    """
    monitor = MultiGPUMonitor()
    snapshot = monitor.snapshot()
    
    if snapshot is None:
        return None
    
    return [
        {
            'device_id': d.device_id,
            'gpu_util': d.gpu_util_percent,
            'memory_util': d.memory_util_percent,
        }
        for d in snapshot.devices
    ]


def check_gpu_health() -> Tuple[bool, str]:
    """Quick health check for all GPUs.
    
    Returns:
        Tuple of (is_healthy, message).
        is_healthy is False if there's a severe imbalance.
        
    Example:
        >>> healthy, msg = check_gpu_health()
        >>> if not healthy:
        ...     print(f"GPU issue: {msg}")
    """
    monitor = MultiGPUMonitor()
    report = monitor.check_imbalance()
    
    if report is None:
        return True, "GPU monitoring unavailable"
    
    if report.severity == "severe":
        return False, report.message
    
    return True, f"GPUs healthy. Avg utilization: {report.avg_utilization:.0f}%"
