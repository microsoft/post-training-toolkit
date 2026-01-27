
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from post_training_toolkit.models.distributed.rank import (
    get_rank,
    get_world_size,
    get_local_rank,
    is_distributed,
    is_initialized,
    is_main_process,
)
from post_training_toolkit.models.distributed.aggregation import (
    gather_scalar,
    all_gather_object,
)

def _get_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None

def _get_gpu_memory_mb() -> Optional[float]:
    torch = _get_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    
    try:
        local_rank = get_local_rank()
        allocated = torch.cuda.memory_allocated(local_rank)
        return allocated / (1024 * 1024)
    except Exception:
        return None

def _get_gpu_memory_reserved_mb() -> Optional[float]:
    torch = _get_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    
    try:
        local_rank = get_local_rank()
        reserved = torch.cuda.memory_reserved(local_rank)
        return reserved / (1024 * 1024)
    except Exception:
        return None

def _get_gpu_max_memory_mb() -> Optional[float]:
    torch = _get_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    
    try:
        local_rank = get_local_rank()
        max_allocated = torch.cuda.max_memory_allocated(local_rank)
        return max_allocated / (1024 * 1024)
    except Exception:
        return None

def _get_gpu_total_memory_mb() -> Optional[float]:
    torch = _get_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    
    try:
        local_rank = get_local_rank()
        props = torch.cuda.get_device_properties(local_rank)
        return props.total_memory / (1024 * 1024)
    except Exception:
        return None

@dataclass
class DistributedMemorySnapshot:
    
    per_rank_allocated_mb: List[float]
    per_rank_reserved_mb: List[float]
    
    mean_mb: float
    max_mb: float
    min_mb: float
    total_mb: float
    
    max_rank: int
    min_rank: int
    
    imbalance_ratio: float
    
    world_size: int
    total_device_memory_mb: Optional[float] = None
    
    @property
    def is_imbalanced(self) -> bool:
        return self.imbalance_ratio > 0.20
    
    @property
    def utilization_ratio(self) -> Optional[float]:
        if self.total_device_memory_mb is None or self.total_device_memory_mb == 0:
            return None
        return self.max_mb / self.total_device_memory_mb
    
    def format(self) -> str:
        lines = [
            f"Distributed Memory Snapshot (world_size={self.world_size})",
            "=" * 50,
            f"  Mean: {self.mean_mb:,.0f} MB",
            f"  Max:  {self.max_mb:,.0f} MB (rank {self.max_rank})",
            f"  Min:  {self.min_mb:,.0f} MB (rank {self.min_rank})",
            f"  Imbalance: {self.imbalance_ratio:.1%}",
        ]
        
        if self.utilization_ratio is not None:
            lines.append(f"  Utilization: {self.utilization_ratio:.1%} of {self.total_device_memory_mb:,.0f} MB")
        
        if self.is_imbalanced:
            lines.append(f"  ⚠️  Memory imbalanced! Rank {self.max_rank} using significantly more.")
        
        return "\n".join(lines)

def get_distributed_memory_snapshot() -> DistributedMemorySnapshot:
    local_allocated = _get_gpu_memory_mb() or 0.0
    local_reserved = _get_gpu_memory_reserved_mb() or 0.0
    total_device = _get_gpu_total_memory_mb()
    
    world_size = get_world_size()
    
    if not is_distributed() or not is_initialized():
        return DistributedMemorySnapshot(
            per_rank_allocated_mb=[local_allocated],
            per_rank_reserved_mb=[local_reserved],
            mean_mb=local_allocated,
            max_mb=local_allocated,
            min_mb=local_allocated,
            total_mb=local_allocated,
            max_rank=0,
            min_rank=0,
            imbalance_ratio=0.0,
            world_size=1,
            total_device_memory_mb=total_device,
        )
    
    local_data = {
        "allocated": local_allocated,
        "reserved": local_reserved,
        "rank": get_rank(),
    }
    all_data = all_gather_object(local_data)
    
    per_rank_allocated = [d["allocated"] for d in all_data]
    per_rank_reserved = [d["reserved"] for d in all_data]
    
    mean_mb = sum(per_rank_allocated) / len(per_rank_allocated)
    max_mb = max(per_rank_allocated)
    min_mb = min(per_rank_allocated)
    total_mb = sum(per_rank_allocated)
    
    max_rank = per_rank_allocated.index(max_mb)
    min_rank = per_rank_allocated.index(min_mb)
    
    imbalance_ratio = (max_mb - min_mb) / mean_mb if mean_mb > 0 else 0.0
    
    return DistributedMemorySnapshot(
        per_rank_allocated_mb=per_rank_allocated,
        per_rank_reserved_mb=per_rank_reserved,
        mean_mb=mean_mb,
        max_mb=max_mb,
        min_mb=min_mb,
        total_mb=total_mb,
        max_rank=max_rank,
        min_rank=min_rank,
        imbalance_ratio=imbalance_ratio,
        world_size=world_size,
        total_device_memory_mb=total_device,
    )

@dataclass
class DistributedMemoryReport:
    
    current_snapshot: DistributedMemorySnapshot
    
    initial_mb_per_rank: List[float]
    current_mb_per_rank: List[float]
    growth_mb_per_rank: List[float]
    
    highest_growth_rank: int
    highest_growth_mb: float
    
    predicted_oom_rank: Optional[int] = None
    steps_until_oom: Optional[int] = None
    
    def format(self) -> str:
        lines = [
            "Distributed Memory Report",
            "=" * 50,
            "",
            self.current_snapshot.format(),
            "",
            "Memory Growth:",
            f"  Highest growth: Rank {self.highest_growth_rank} (+{self.highest_growth_mb:,.0f} MB)",
        ]
        
        if self.predicted_oom_rank is not None:
            lines.append(f"  ⚠️  Rank {self.predicted_oom_rank} predicted to OOM first!")
            if self.steps_until_oom is not None:
                lines.append(f"      Estimated steps until OOM: ~{self.steps_until_oom}")
        
        return "\n".join(lines)

class DistributedMemoryTracker:
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.snapshots: List[DistributedMemorySnapshot] = []
        self.steps: List[int] = []
        
        self._initial_snapshot: Optional[DistributedMemorySnapshot] = None
    
    def record(self, step: int) -> DistributedMemorySnapshot:
        snapshot = get_distributed_memory_snapshot()
        
        if self._initial_snapshot is None:
            self._initial_snapshot = snapshot
        
        self.snapshots.append(snapshot)
        self.steps.append(step)
        
        if len(self.snapshots) > self.history_size:
            self.snapshots = self.snapshots[-self.history_size:]
            self.steps = self.steps[-self.history_size:]
        
        return snapshot
    
    def has_memory_issue(self, imbalance_threshold: float = 0.20) -> bool:
        if not self.snapshots:
            return False
        
        latest = self.snapshots[-1]
        
        if latest.imbalance_ratio > imbalance_threshold:
            return True
        
        if latest.utilization_ratio is not None and latest.utilization_ratio > 0.90:
            return True
        
        if self._initial_snapshot is not None and len(self.snapshots) > 10:
            initial_max = self._initial_snapshot.max_mb
            current_max = latest.max_mb
            growth_ratio = (current_max - initial_max) / initial_max if initial_max > 0 else 0
            if growth_ratio > 0.50:
                return True
        
        return False
    
    def report(self) -> DistributedMemoryReport:
        if not self.snapshots:
            empty_snapshot = get_distributed_memory_snapshot()
            return DistributedMemoryReport(
                current_snapshot=empty_snapshot,
                initial_mb_per_rank=empty_snapshot.per_rank_allocated_mb,
                current_mb_per_rank=empty_snapshot.per_rank_allocated_mb,
                growth_mb_per_rank=[0.0] * empty_snapshot.world_size,
                highest_growth_rank=0,
                highest_growth_mb=0.0,
            )
        
        current = self.snapshots[-1]
        initial = self._initial_snapshot or self.snapshots[0]
        
        growth_per_rank = [
            current.per_rank_allocated_mb[i] - initial.per_rank_allocated_mb[i]
            for i in range(current.world_size)
        ]
        
        highest_growth_mb = max(growth_per_rank)
        highest_growth_rank = growth_per_rank.index(highest_growth_mb)
        
        predicted_oom_rank = None
        steps_until_oom = None
        
        if current.total_device_memory_mb is not None and len(self.steps) > 10:
            total_steps = self.steps[-1] - self.steps[0]
            if total_steps > 0:
                for rank in range(current.world_size):
                    current_mb = current.per_rank_allocated_mb[rank]
                    growth_mb = growth_per_rank[rank]
                    
                    if growth_mb > 0:
                        remaining_mb = current.total_device_memory_mb * 0.95 - current_mb
                        growth_rate = growth_mb / total_steps
                        
                        if growth_rate > 0:
                            steps_to_oom = int(remaining_mb / growth_rate)
                            
                            if steps_until_oom is None or steps_to_oom < steps_until_oom:
                                steps_until_oom = steps_to_oom
                                predicted_oom_rank = rank
        
        return DistributedMemoryReport(
            current_snapshot=current,
            initial_mb_per_rank=initial.per_rank_allocated_mb,
            current_mb_per_rank=current.per_rank_allocated_mb,
            growth_mb_per_rank=growth_per_rank,
            highest_growth_rank=highest_growth_rank,
            highest_growth_mb=highest_growth_mb,
            predicted_oom_rank=predicted_oom_rank,
            steps_until_oom=steps_until_oom,
        )
