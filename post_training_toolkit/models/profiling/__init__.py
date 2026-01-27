"""Profiling module for training performance analysis.

Provides automated detection of:
- Training slowdowns (step time degradation)
- Throughput bottlenecks (tokens/sec, GPU utilization)
- GIL contention in Python code
- Memory pressure and leaks
- Multi-GPU utilization imbalance

All profiling is automatically integrated into DiagnosticsCallback.
Users don't need to import anything from this module directly.
"""

from post_training_toolkit.models.profiling.step_timer import StepTimer, SlowdownDetector
from post_training_toolkit.models.profiling.throughput import ThroughputTracker
from post_training_toolkit.models.profiling.gil import GILContention, detect_gil_contention
from post_training_toolkit.models.profiling.gpu import (
    GPUProfiler,
    MultiGPUMonitor,
    MultiGPUSnapshot,
    GPUDeviceStatus,
    GPUImbalanceReport,
    get_all_gpu_utilization,
    check_gpu_health,
)

__all__ = [
    "StepTimer",
    "SlowdownDetector", 
    "ThroughputTracker",
    "GILContention",
    "detect_gil_contention",
    "GPUProfiler",
    # Multi-GPU monitoring
    "MultiGPUMonitor",
    "MultiGPUSnapshot",
    "GPUDeviceStatus",
    "GPUImbalanceReport",
    "get_all_gpu_utilization",
    "check_gpu_health",
]
