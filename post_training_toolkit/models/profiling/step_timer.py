
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque

@dataclass
class StepTiming:
    step: int
    duration_sec: float
    timestamp: float
    memory_mb: Optional[float] = None

@dataclass
class SlowdownEvent:
    step: int
    current_duration: float
    baseline_duration: float
    slowdown_factor: float
    likely_cause: str
    suggestion: str
    memory_growth_mb: Optional[float] = None

class StepTimer:
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.timings: List[StepTiming] = []
        self._current_step: Optional[int] = None
        self._step_start: Optional[float] = None
        
    def start_step(self, step: int) -> None:
        self._current_step = step
        self._step_start = time.perf_counter()
        
    def end_step(self, memory_mb: Optional[float] = None) -> Optional[StepTiming]:
        if self._step_start is None or self._current_step is None:
            return None
            
        duration = time.perf_counter() - self._step_start
        timing = StepTiming(
            step=self._current_step,
            duration_sec=duration,
            timestamp=time.time(),
            memory_mb=memory_mb,
        )
        self.timings.append(timing)
        
        self._current_step = None
        self._step_start = None
        
        return timing
        
    def get_baseline_duration(self, exclude_recent: int = 5) -> Optional[float]:
        min_required = max(20, self.window_size // 2) + exclude_recent
        if len(self.timings) < min_required:
            return None
            
        warmup = min(5, len(self.timings) // 10)
        baseline_end = min(warmup + self.window_size, len(self.timings) - exclude_recent)
        baseline_timings = self.timings[warmup:baseline_end]
        
        if len(baseline_timings) < 5:
            return None
            
        durations = sorted(t.duration_sec for t in baseline_timings)
        mid = len(durations) // 2
        return durations[mid]
        
    def get_recent_duration(self, window: int = 10) -> Optional[float]:
        if len(self.timings) < window:
            return None
            
        recent = self.timings[-window:]
        durations = sorted(t.duration_sec for t in recent)
        mid = len(durations) // 2
        return durations[mid]
        
    def get_memory_growth(self) -> Optional[float]:
        memory_readings = [t.memory_mb for t in self.timings if t.memory_mb is not None]
        if len(memory_readings) < 2:
            return None
            
        early = sum(memory_readings[:10]) / min(10, len(memory_readings))
        recent = sum(memory_readings[-10:]) / min(10, len(memory_readings))
        return recent - early
        
    @property
    def total_steps(self) -> int:
        return len(self.timings)
        
    @property
    def total_time_sec(self) -> float:
        return sum(t.duration_sec for t in self.timings)
        
    def summary(self) -> dict:
        if not self.timings:
            return {"total_steps": 0}
            
        durations = [t.duration_sec for t in self.timings]
        return {
            "total_steps": len(self.timings),
            "total_time_sec": sum(durations),
            "mean_step_sec": sum(durations) / len(durations),
            "min_step_sec": min(durations),
            "max_step_sec": max(durations),
            "baseline_duration": self.get_baseline_duration(),
            "recent_duration": self.get_recent_duration(),
            "memory_growth_mb": self.get_memory_growth(),
        }

class SlowdownDetector:
    
    def __init__(
        self,
        threshold: float = 1.5,
        severe_threshold: float = 2.0,
        min_steps_for_baseline: int = 50,
        check_interval: int = 10,
    ):
        self.threshold = threshold
        self.severe_threshold = severe_threshold
        self.min_steps_for_baseline = min_steps_for_baseline
        self.check_interval = check_interval
        
        self._last_check_step = 0
        self._events: List[SlowdownEvent] = []
        
    def check(self, timer: StepTimer) -> Optional[SlowdownEvent]:
        if timer.total_steps < self.min_steps_for_baseline:
            return None
        if timer.total_steps - self._last_check_step < self.check_interval:
            return None
            
        self._last_check_step = timer.total_steps
        
        baseline = timer.get_baseline_duration()
        recent = timer.get_recent_duration()
        
        if baseline is None or recent is None:
            return None
            
        slowdown_factor = recent / baseline
        
        if slowdown_factor < self.threshold:
            return None
            
        memory_growth = timer.get_memory_growth()
        likely_cause, suggestion = self._diagnose(slowdown_factor, memory_growth)
        
        event = SlowdownEvent(
            step=timer.total_steps,
            current_duration=recent,
            baseline_duration=baseline,
            slowdown_factor=slowdown_factor,
            likely_cause=likely_cause,
            suggestion=suggestion,
            memory_growth_mb=memory_growth,
        )
        self._events.append(event)
        return event
        
    def _diagnose(
        self, 
        slowdown_factor: float, 
        memory_growth_mb: Optional[float]
    ) -> Tuple[str, str]:
        if memory_growth_mb is not None and memory_growth_mb > 1000:
            return (
                "Memory pressure (likely fragmentation or leak)",
                "Try: gradient_checkpointing=True, reduce batch size, or restart process between runs"
            )
            
        if memory_growth_mb is not None and memory_growth_mb > 500:
            return (
                "Moderate memory growth",
                "Monitor for continued growth. Consider enabling gradient checkpointing."
            )
            
        if slowdown_factor > self.severe_threshold:
            return (
                "Severe slowdown (possible I/O or system issue)",
                "Check: disk I/O, other processes, thermal throttling, dataloader workers"
            )
            
        return (
            "Gradual training slowdown",
            "Common causes: memory fragmentation, dataset iteration overhead, accumulated state"
        )
        
    @property
    def events(self) -> List[SlowdownEvent]:
        return self._events.copy()
        
    @property
    def has_slowdown(self) -> bool:
        return len(self._events) > 0
        
    def worst_slowdown(self) -> Optional[SlowdownEvent]:
        if not self._events:
            return None
        return max(self._events, key=lambda e: e.slowdown_factor)
        
    def summary(self) -> dict:
        worst = self.worst_slowdown()
        return {
            "slowdown_detected": self.has_slowdown,
            "num_events": len(self._events),
            "worst_factor": worst.slowdown_factor if worst else None,
            "worst_cause": worst.likely_cause if worst else None,
        }
