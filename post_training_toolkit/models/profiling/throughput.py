"""Throughput tracking for training performance.

Measures:
- Tokens per second
- Samples per second  
- GPU utilization efficiency
- Dataloader vs compute balance

Helps identify whether training is compute-bound, memory-bound, or I/O-bound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque


@dataclass
class ThroughputSample:
    """Single throughput measurement."""
    step: int
    tokens_per_sec: Optional[float] = None
    samples_per_sec: Optional[float] = None
    duration_sec: float = 0.0
    batch_size: Optional[int] = None
    seq_length: Optional[int] = None


@dataclass 
class ThroughputReport:
    """Aggregated throughput report."""
    mean_tokens_per_sec: Optional[float]
    mean_samples_per_sec: Optional[float]
    peak_tokens_per_sec: Optional[float]
    total_tokens: int
    total_samples: int
    total_time_sec: float
    efficiency_estimate: Optional[float]  # 0-1, how close to theoretical max
    bottleneck: Optional[str]  # "compute", "memory", "io", or None
    

class ThroughputTracker:
    """Tracks training throughput over time.
    
    Usage:
        tracker = ThroughputTracker()
        
        for step, batch in enumerate(dataloader):
            tracker.start_step()
            # ... training ...
            tracker.end_step(
                num_tokens=batch_size * seq_len,
                num_samples=batch_size
            )
            
        print(tracker.report())
    """
    
    def __init__(self, window_size: int = 100):
        """Initialize throughput tracker.
        
        Args:
            window_size: Window for moving average calculations
        """
        self.window_size = window_size
        self.samples: List[ThroughputSample] = []
        
        self._step_start: Optional[float] = None
        self._current_step: int = 0
        
        # Running totals
        self._total_tokens: int = 0
        self._total_samples: int = 0
        self._total_time: float = 0.0
        
        # For efficiency estimation
        self._model_params: Optional[int] = None
        self._theoretical_max_tps: Optional[float] = None
        
    def set_model_info(
        self, 
        num_params: Optional[int] = None,
        theoretical_max_tps: Optional[float] = None
    ) -> None:
        """Set model info for efficiency calculations.
        
        Args:
            num_params: Number of model parameters
            theoretical_max_tps: Known theoretical max tokens/sec for this setup
        """
        self._model_params = num_params
        self._theoretical_max_tps = theoretical_max_tps
        
    def start_step(self) -> None:
        """Mark start of a training step."""
        self._step_start = time.perf_counter()
        
    def end_step(
        self,
        num_tokens: Optional[int] = None,
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
    ) -> ThroughputSample:
        """Mark end of training step and record throughput.
        
        Args:
            num_tokens: Total tokens processed this step
            num_samples: Number of samples/sequences processed
            batch_size: Batch size (alternative to num_samples)
            seq_length: Sequence length (used with batch_size to calc tokens)
            
        Returns:
            ThroughputSample for this step
        """
        if self._step_start is None:
            # Auto-start if not explicitly started
            duration = 0.0
        else:
            duration = time.perf_counter() - self._step_start
            
        # Calculate num_samples if not provided
        if num_samples is None and batch_size is not None:
            num_samples = batch_size
            
        # Calculate num_tokens if not provided
        if num_tokens is None and batch_size is not None and seq_length is not None:
            num_tokens = batch_size * seq_length
            
        # Calculate rates
        tokens_per_sec = None
        samples_per_sec = None
        
        if duration > 0:
            if num_tokens is not None:
                tokens_per_sec = num_tokens / duration
            if num_samples is not None:
                samples_per_sec = num_samples / duration
                
        sample = ThroughputSample(
            step=self._current_step,
            tokens_per_sec=tokens_per_sec,
            samples_per_sec=samples_per_sec,
            duration_sec=duration,
            batch_size=batch_size,
            seq_length=seq_length,
        )
        self.samples.append(sample)
        
        # Update totals
        if num_tokens:
            self._total_tokens += num_tokens
        if num_samples:
            self._total_samples += num_samples
        self._total_time += duration
        self._current_step += 1
        
        self._step_start = None
        return sample
        
    def get_recent_throughput(self, window: Optional[int] = None) -> Dict[str, Optional[float]]:
        """Get average throughput over recent steps.
        
        Args:
            window: Number of recent steps (default: self.window_size)
            
        Returns:
            Dict with tokens_per_sec and samples_per_sec
        """
        window = window or self.window_size
        recent = self.samples[-window:] if self.samples else []
        
        token_rates = [s.tokens_per_sec for s in recent if s.tokens_per_sec is not None]
        sample_rates = [s.samples_per_sec for s in recent if s.samples_per_sec is not None]
        
        return {
            "tokens_per_sec": sum(token_rates) / len(token_rates) if token_rates else None,
            "samples_per_sec": sum(sample_rates) / len(sample_rates) if sample_rates else None,
        }
        
    def detect_bottleneck(self) -> Optional[str]:
        """Attempt to detect training bottleneck.
        
        Returns:
            "compute", "memory", "io", or None if can't determine
        """
        if len(self.samples) < 20:
            return None
            
        # Look at throughput variance - high variance often indicates I/O issues
        recent = self.samples[-50:]
        token_rates = [s.tokens_per_sec for s in recent if s.tokens_per_sec]
        
        if not token_rates:
            return None
            
        mean_rate = sum(token_rates) / len(token_rates)
        variance = sum((r - mean_rate) ** 2 for r in token_rates) / len(token_rates)
        cv = (variance ** 0.5) / mean_rate if mean_rate > 0 else 0  # Coefficient of variation
        
        # High variance suggests I/O or dataloader issues
        if cv > 0.3:
            return "io"
            
        # If we have theoretical max, we can estimate compute vs memory bound
        if self._theoretical_max_tps is not None:
            efficiency = mean_rate / self._theoretical_max_tps
            if efficiency < 0.5:
                return "memory"  # Not hitting compute limits, likely memory bound
            elif efficiency > 0.8:
                return "compute"  # Close to theoretical max
                
        return None
        
    def report(self) -> ThroughputReport:
        """Generate throughput report."""
        if not self.samples:
            return ThroughputReport(
                mean_tokens_per_sec=None,
                mean_samples_per_sec=None,
                peak_tokens_per_sec=None,
                total_tokens=0,
                total_samples=0,
                total_time_sec=0.0,
                efficiency_estimate=None,
                bottleneck=None,
            )
            
        token_rates = [s.tokens_per_sec for s in self.samples if s.tokens_per_sec is not None]
        sample_rates = [s.samples_per_sec for s in self.samples if s.samples_per_sec is not None]
        
        mean_tps = sum(token_rates) / len(token_rates) if token_rates else None
        peak_tps = max(token_rates) if token_rates else None
        
        # Calculate efficiency
        efficiency = None
        if mean_tps and self._theoretical_max_tps:
            efficiency = mean_tps / self._theoretical_max_tps
            
        return ThroughputReport(
            mean_tokens_per_sec=mean_tps,
            mean_samples_per_sec=sum(sample_rates) / len(sample_rates) if sample_rates else None,
            peak_tokens_per_sec=peak_tps,
            total_tokens=self._total_tokens,
            total_samples=self._total_samples,
            total_time_sec=self._total_time,
            efficiency_estimate=efficiency,
            bottleneck=self.detect_bottleneck(),
        )
        
    def format_report(self) -> str:
        """Format throughput report as human-readable string."""
        r = self.report()
        
        lines = ["Throughput Report", "=" * 40]
        
        if r.mean_tokens_per_sec:
            lines.append(f"Mean throughput: {r.mean_tokens_per_sec:,.0f} tokens/sec")
        if r.peak_tokens_per_sec:
            lines.append(f"Peak throughput: {r.peak_tokens_per_sec:,.0f} tokens/sec")
        if r.mean_samples_per_sec:
            lines.append(f"Sample rate: {r.mean_samples_per_sec:.1f} samples/sec")
            
        lines.append(f"Total tokens: {r.total_tokens:,}")
        lines.append(f"Total time: {r.total_time_sec:.1f}s")
        
        if r.efficiency_estimate:
            pct = r.efficiency_estimate * 100
            lines.append(f"Efficiency: {pct:.0f}% of theoretical max")
            
        if r.bottleneck:
            bottleneck_msgs = {
                "io": "⚠️ Bottleneck: I/O (dataloader or disk)",
                "memory": "⚠️ Bottleneck: Memory bandwidth",
                "compute": "✓ Compute-bound (efficient)",
            }
            lines.append(bottleneck_msgs.get(r.bottleneck, f"Bottleneck: {r.bottleneck}"))
            
        return "\n".join(lines)
