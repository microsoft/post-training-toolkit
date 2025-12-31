"""Distributed training support for multi-GPU/multi-node profiling.

This module provides utilities for monitoring and profiling distributed training
runs across multiple GPUs and nodes. It works with any distributed backend
(DDP, FSDP, DeepSpeed) by using the existing torch.distributed setup.

Key capabilities:
- Rank-aware logging (only rank 0 writes files)
- Cross-rank metric aggregation (mean, max, min across all ranks)
- Straggler detection (identify which rank is slowing down training)
- Distributed memory tracking (compare GPU memory across ranks)

Usage:
    from post_training_toolkit.models.distributed import (
        get_rank,
        is_main_process,
        gather_dict,
        StragglerDetector,
    )
    
    # Check if we should log
    if is_main_process():
        save_checkpoint()
    
    # Aggregate metrics across ranks
    local_metrics = {"step_time": 0.8, "memory_mb": 15000}
    global_metrics = gather_dict(local_metrics)
    # Returns: {"step_time_mean": 0.85, "step_time_max": 1.2, ...}
    
    # Detect stragglers
    detector = StragglerDetector()
    detector.record_step(step=100, duration=local_step_time)
    report = detector.analyze()
    if report and report.has_straggler:
        print(f"Rank {report.slowest_rank} is bottleneck")

Note:
    This module does NOT implement distributed training itself—it monitors
    training that's already running in FSDP/DDP/DeepSpeed. It uses the same
    torch.distributed backend that those frameworks initialize.
"""

from post_training_toolkit.models.distributed.rank import (
    get_rank,
    get_local_rank,
    get_world_size,
    is_main_process,
    is_distributed,
    is_initialized,
    get_backend,
    get_device,
    barrier,
    DistributedInfo,
    get_distributed_info,
)
from post_training_toolkit.models.distributed.aggregation import (
    gather_scalar,
    gather_dict,
    all_gather_object,
    broadcast_object,
    reduce_tensor,
)
from post_training_toolkit.models.distributed.memory import (
    DistributedMemoryTracker,
    DistributedMemorySnapshot,
    get_distributed_memory_snapshot,
)
from post_training_toolkit.models.distributed.straggler import (
    StragglerDetector,
    StragglerReport,
)

__all__ = [
    # rank.py
    "get_rank",
    "get_local_rank",
    "get_world_size",
    "is_main_process",
    "is_distributed",
    "is_initialized",
    "get_backend",
    "get_device",
    "barrier",
    "DistributedInfo",
    "get_distributed_info",
    # aggregation.py
    "gather_scalar",
    "gather_dict",
    "all_gather_object",
    "broadcast_object",
    "reduce_tensor",
    # memory.py
    "DistributedMemoryTracker",
    "DistributedMemorySnapshot",
    "get_distributed_memory_snapshot",
    # straggler.py
    "StragglerDetector",
    "StragglerReport",
]
