
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from contextlib import contextmanager

@dataclass
class GILContention:
    total_wait_time_sec: float
    total_measured_time_sec: float
    contention_ratio: float
    num_contentions: int
    hotspots: List[str]
    
    @property
    def is_significant(self) -> bool:
        return self.contention_ratio > 0.10
        
    @property
    def is_severe(self) -> bool:
        return self.contention_ratio > 0.25
        
    def format(self) -> str:
        pct = self.contention_ratio * 100
        severity = "SEVERE" if self.is_severe else "SIGNIFICANT" if self.is_significant else "LOW"
        
        lines = [
            f"GIL Contention: {pct:.1f}% ({severity})",
            f"  Wait time: {self.total_wait_time_sec:.2f}s / {self.total_measured_time_sec:.2f}s",
            f"  Contention events: {self.num_contentions}",
        ]
        
        if self.hotspots:
            lines.append("  Hotspots:")
            for h in self.hotspots[:5]:
                lines.append(f"    - {h}")
                
        return "\n".join(lines)

class GILMonitor:
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self._samples: List[Tuple[float, bool]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        
        self._operation_times: Dict[str, List[float]] = {}
        self._current_operation: Optional[str] = None
        
    def start(self) -> None:
        if self._running:
            return
            
        self._running = True
        self._start_time = time.perf_counter()
        self._samples.clear()
        
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        
    def stop(self) -> None:
        self._running = False
        self._stop_time = time.perf_counter()
        
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
            
    def _sample_loop(self) -> None:
        while self._running:
            start = time.perf_counter()
            
            time.sleep(0.001)
            
            elapsed = time.perf_counter() - start
            was_contended = elapsed > 0.005
            
            self._samples.append((time.perf_counter(), was_contended))
            time.sleep(self.sample_interval)
            
    @contextmanager
    def track_operation(self, name: str):
        start = time.perf_counter()
        prev_operation = self._current_operation
        self._current_operation = name
        
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self._operation_times:
                self._operation_times[name] = []
            self._operation_times[name].append(elapsed)
            self._current_operation = prev_operation
            
    def analyze(self) -> GILContention:
        if not self._samples:
            return GILContention(
                total_wait_time_sec=0.0,
                total_measured_time_sec=0.0,
                contention_ratio=0.0,
                num_contentions=0,
                hotspots=[],
            )
            
        num_contended = sum(1 for _, contended in self._samples if contended)
        total_time = (self._stop_time or time.perf_counter()) - (self._start_time or 0)
        
        estimated_wait = num_contended * self.sample_interval * 0.5
        
        hotspots = []
        for name, times in sorted(
            self._operation_times.items(), 
            key=lambda x: sum(x[1]), 
            reverse=True
        ):
            total = sum(times)
            avg = total / len(times) if times else 0
            if avg > 0.01:
                hotspots.append(f"{name}: {avg*1000:.1f}ms avg ({len(times)} calls)")
                
        return GILContention(
            total_wait_time_sec=estimated_wait,
            total_measured_time_sec=total_time,
            contention_ratio=estimated_wait / total_time if total_time > 0 else 0,
            num_contentions=num_contended,
            hotspots=hotspots,
        )

def detect_gil_contention(
    func: Callable,
    *args,
    duration_sec: float = 10.0,
    **kwargs
) -> Tuple[any, GILContention]:
    monitor = GILMonitor(sample_interval=0.05)
    monitor.start()
    
    try:
        result = func(*args, **kwargs)
    finally:
        monitor.stop()
        
    return result, monitor.analyze()

class DataloaderGILProfiler:
    
    def __init__(self):
        self._batch_times: List[float] = []
        self._batch_start: Optional[float] = None
        self._inter_batch_times: List[float] = []
        self._last_batch_end: Optional[float] = None
        
    @contextmanager
    def track_batch(self):
        if self._last_batch_end is not None:
            inter_batch = time.perf_counter() - self._last_batch_end
            self._inter_batch_times.append(inter_batch)
            
        self._batch_start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - self._batch_start
            self._batch_times.append(elapsed)
            self._last_batch_end = time.perf_counter()
            
    def report(self) -> Dict[str, float]:
        if not self._batch_times:
            return {"error": "No batches tracked"}
            
        avg_batch = sum(self._batch_times) / len(self._batch_times)
        avg_inter = sum(self._inter_batch_times) / len(self._inter_batch_times) if self._inter_batch_times else 0
        
        dataloader_ratio = avg_inter / (avg_batch + avg_inter) if (avg_batch + avg_inter) > 0 else 0
        
        report = {
            "avg_batch_time_ms": avg_batch * 1000,
            "avg_dataloader_time_ms": avg_inter * 1000,
            "dataloader_ratio": dataloader_ratio,
            "total_batches": len(self._batch_times),
        }
        
        if dataloader_ratio > 0.5:
            report["recommendation"] = (
                "Dataloader taking >50% of time. Consider: "
                "num_workers > 0, pin_memory=True, simpler collate_fn"
            )
        elif dataloader_ratio > 0.25:
            report["recommendation"] = (
                "Dataloader taking 25-50% of time. May benefit from "
                "increasing num_workers or optimizing collate function."
            )
        else:
            report["recommendation"] = "Dataloader efficiency looks good."
            
        return report
