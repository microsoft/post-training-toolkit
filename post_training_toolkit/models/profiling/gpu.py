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
