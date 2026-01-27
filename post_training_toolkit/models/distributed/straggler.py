
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from post_training_toolkit.models.distributed.rank import (
    get_rank,
    get_world_size,
    is_distributed,
    is_initialized,
    is_main_process,
)
from post_training_toolkit.models.distributed.aggregation import (
    all_gather_object,
    gather_scalar,
)

@dataclass
class StragglerReport:
    
    has_straggler: bool
    
    slowest_rank: int
    slowdown_factor: float
    
    is_consistent: bool
    consecutive_slow_checks: int
    
    per_rank_mean_sec: List[float]
    per_rank_last_sec: List[float]
    
    mean_step_sec: float
    max_step_sec: float
    min_step_sec: float
    
    likely_cause: str
    suggestion: str
    
    world_size: int
    steps_analyzed: int
    
    def format(self) -> str:
        lines = [
            "Straggler Analysis Report",
            "=" * 50,
            f"World size: {self.world_size}",
            f"Steps analyzed: {self.steps_analyzed}",
            "",
            f"Mean step time: {self.mean_step_sec*1000:.1f} ms",
            f"Fastest rank:   {self.min_step_sec*1000:.1f} ms",
            f"Slowest rank:   {self.max_step_sec*1000:.1f} ms (rank {self.slowest_rank})",
            "",
        ]
        
        if self.has_straggler:
            lines.extend([
                f"⚠️  STRAGGLER DETECTED: Rank {self.slowest_rank}",
                f"    Slowdown factor: {self.slowdown_factor:.2f}x mean",
                f"    Consistent: {'Yes' if self.is_consistent else 'No'} ({self.consecutive_slow_checks} consecutive checks)",
                f"    Likely cause: {self.likely_cause}",
                f"    Suggestion: {self.suggestion}",
            ])
        else:
            lines.append("✓ No significant straggler detected")
        
        lines.extend([
            "",
            "Per-rank mean step time:",
        ])
        for rank, mean_sec in enumerate(self.per_rank_mean_sec):
            marker = " ← slowest" if rank == self.slowest_rank and self.has_straggler else ""
            lines.append(f"  Rank {rank}: {mean_sec*1000:.1f} ms{marker}")
        
        return "\n".join(lines)

class StragglerDetector:
    
    def __init__(
        self,
        window_size: int = 50,
        straggler_threshold: float = 1.20,
        consistent_checks: int = 3,
    ):
        self.window_size = window_size
        self.straggler_threshold = straggler_threshold
        self.consistent_checks = consistent_checks
        
        self._step_times: Deque[float] = deque(maxlen=window_size)
        self._steps: Deque[int] = deque(maxlen=window_size)
        self._step_start: Optional[float] = None
        
        self._slowest_rank_history: Deque[int] = deque(maxlen=10)
        
        self._memory_at_step: Dict[int, float] = {}
    
    def start_step(self) -> None:
        self._step_start = time.perf_counter()
    
    def end_step(self, step: int, duration: Optional[float] = None) -> float:
        if duration is None:
            if self._step_start is None:
                raise ValueError("Must call start_step() before end_step() if duration not provided")
            duration = time.perf_counter() - self._step_start
            self._step_start = None
        
        self._step_times.append(duration)
        self._steps.append(step)
        
        return duration
    
    def record_step(self, step: int, duration: float, memory_mb: Optional[float] = None) -> None:
        self._step_times.append(duration)
        self._steps.append(step)
        
        if memory_mb is not None:
            self._memory_at_step[step] = memory_mb
    
    def analyze(self) -> Optional[StragglerReport]:
        if len(self._step_times) < 5:
            return None
        
        local_times = list(self._step_times)
        local_mean = sum(local_times) / len(local_times)
        local_last = local_times[-1] if local_times else 0.0
        
        world_size = get_world_size()
        
        if not is_distributed() or not is_initialized():
            return StragglerReport(
                has_straggler=False,
                slowest_rank=0,
                slowdown_factor=1.0,
                is_consistent=False,
                consecutive_slow_checks=0,
                per_rank_mean_sec=[local_mean],
                per_rank_last_sec=[local_last],
                mean_step_sec=local_mean,
                max_step_sec=local_mean,
                min_step_sec=local_mean,
                likely_cause="N/A",
                suggestion="N/A",
                world_size=1,
                steps_analyzed=len(local_times),
            )
        
        local_data = {
            "rank": get_rank(),
            "mean_sec": local_mean,
            "last_sec": local_last,
            "num_steps": len(local_times),
        }
        all_data = all_gather_object(local_data)
        
        all_data = sorted(all_data, key=lambda x: x["rank"])
        
        per_rank_mean = [d["mean_sec"] for d in all_data]
        per_rank_last = [d["last_sec"] for d in all_data]
        
        mean_step = sum(per_rank_mean) / len(per_rank_mean)
        max_step = max(per_rank_mean)
        min_step = min(per_rank_mean)
        
        slowest_rank = per_rank_mean.index(max_step)
        slowdown_factor = max_step / mean_step if mean_step > 0 else 1.0
        
        self._slowest_rank_history.append(slowest_rank)
        
        recent_slowest = list(self._slowest_rank_history)[-self.consistent_checks:]
        is_consistent = (
            len(recent_slowest) >= self.consistent_checks
            and all(r == slowest_rank for r in recent_slowest)
        )
        
        consecutive = 0
        for r in reversed(list(self._slowest_rank_history)):
            if r == slowest_rank:
                consecutive += 1
            else:
                break
        
        has_straggler = slowdown_factor >= self.straggler_threshold
        
        likely_cause, suggestion = self._diagnose_cause(
            slowest_rank=slowest_rank,
            slowdown_factor=slowdown_factor,
            is_consistent=is_consistent,
            per_rank_mean=per_rank_mean,
        )
        
        return StragglerReport(
            has_straggler=has_straggler,
            slowest_rank=slowest_rank,
            slowdown_factor=slowdown_factor,
            is_consistent=is_consistent,
            consecutive_slow_checks=consecutive,
            per_rank_mean_sec=per_rank_mean,
            per_rank_last_sec=per_rank_last,
            mean_step_sec=mean_step,
            max_step_sec=max_step,
            min_step_sec=min_step,
            likely_cause=likely_cause,
            suggestion=suggestion,
            world_size=world_size,
            steps_analyzed=len(local_times),
        )
    
    def _diagnose_cause(
        self,
        slowest_rank: int,
        slowdown_factor: float,
        is_consistent: bool,
        per_rank_mean: List[float],
    ) -> Tuple[str, str]:
        has_memory_correlation = False
        if self._memory_at_step:
            pass
        
        if slowdown_factor > 2.0:
            if is_consistent:
                return (
                    "Severe consistent slowdown - likely hardware issue or memory pressure",
                    "Check GPU utilization and memory on rank {slowest_rank}. "
                    "Consider nvidia-smi or profiling tools."
                )
            else:
                return (
                    "Intermittent severe slowdown - likely I/O or GC pause",
                    "Check for checkpoint I/O, logging, or garbage collection. "
                    "Consider staggering checkpoints across ranks."
                )
        
        elif slowdown_factor > 1.5:
            if is_consistent:
                return (
                    "Consistent moderate slowdown - likely data loading imbalance",
                    "Check if dataloader is distributing work evenly. "
                    "Consider profiling the data pipeline on rank {slowest_rank}."
                )
            else:
                return (
                    "Intermittent moderate slowdown - likely callback or logging overhead",
                    "Check if callbacks/logging are only running on rank 0. "
                    "Move non-essential work off the critical path."
                )
        
        elif slowdown_factor > 1.2:
            if is_consistent:
                return (
                    "Slight consistent slowdown - may be hardware variation or thermal throttling",
                    "Usually acceptable. If problematic, check GPU temperatures and clocks."
                )
            else:
                return (
                    "Slight intermittent slowdown - likely normal variation",
                    "Usually not actionable. Monitor for worsening trends."
                )
        
        else:
            return (
                "No significant slowdown detected",
                "Training is well-balanced across ranks."
            )
    
    def get_efficiency(self) -> float:
        if len(self._step_times) < 5 or not is_distributed():
            return 1.0
        
        local_mean = sum(self._step_times) / len(self._step_times)
        
        max_time = gather_scalar(local_mean, op="max")
        mean_time = gather_scalar(local_mean, op="mean")
        
        if max_time == 0:
            return 1.0
        
        return mean_time / max_time
