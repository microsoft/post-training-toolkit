"""Tests for the profiling module."""

import time
import pytest

from post_training_toolkit.models.profiling import (
    StepTimer,
    SlowdownDetector,
    ThroughputTracker,
    GPUProfiler,
)
from post_training_toolkit.models.profiling.gil import GILMonitor, DataloaderGILProfiler


class TestStepTimer:
    """Tests for StepTimer."""
    
    def test_basic_timing(self):
        """Test basic step timing."""
        timer = StepTimer()
        
        timer.start_step(0)
        time.sleep(0.01)  # 10ms
        timing = timer.end_step()
        
        assert timing is not None
        assert timing.step == 0
        assert timing.duration_sec >= 0.01
        assert timing.duration_sec < 0.1  # Shouldn't be too long
        
    def test_timing_accumulation(self):
        """Test that timings accumulate correctly."""
        timer = StepTimer()
        
        for step in range(10):
            timer.start_step(step)
            time.sleep(0.005)  # 5ms
            timer.end_step()
        
        assert timer.total_steps == 10
        assert timer.total_time_sec >= 0.05  # At least 50ms total
        
    def test_memory_tracking(self):
        """Test memory tracking in timings."""
        timer = StepTimer()
        
        timer.start_step(0)
        timing = timer.end_step(memory_mb=1000.0)
        
        assert timing.memory_mb == 1000.0
        
    def test_baseline_calculation(self):
        """Test baseline duration calculation."""
        timer = StepTimer(window_size=10)  # Small window for test
        
        # Add 30 steps with consistent timing (need enough for baseline calculation)
        for step in range(30):
            timer.start_step(step)
            time.sleep(0.005)
            timer.end_step()
        
        baseline = timer.get_baseline_duration()
        assert baseline is not None
        assert baseline >= 0.004  # Should be around 5ms
        assert baseline < 0.02
        
    def test_summary(self):
        """Test summary generation."""
        timer = StepTimer()
        
        for step in range(5):
            timer.start_step(step)
            time.sleep(0.005)
            timer.end_step(memory_mb=1000.0 + step * 10)
        
        summary = timer.summary()
        
        assert summary["total_steps"] == 5
        assert summary["total_time_sec"] >= 0.025
        assert summary["mean_step_sec"] >= 0.005
        assert "min_step_sec" in summary
        assert "max_step_sec" in summary


class TestSlowdownDetector:
    """Tests for SlowdownDetector."""
    
    def test_no_slowdown(self):
        """Test that no slowdown is detected for consistent timing."""
        timer = StepTimer()
        detector = SlowdownDetector(
            threshold=1.5,
            min_steps_for_baseline=10,
            check_interval=5,
        )
        
        # Consistent timing - no slowdown
        for step in range(50):
            timer.start_step(step)
            time.sleep(0.005)
            timer.end_step()
            
            event = detector.check(timer)
            # Should not detect slowdown with consistent timing
            if step > 20:  # After baseline established
                assert event is None or event.slowdown_factor < 1.5
                
    def test_detects_slowdown(self):
        """Test that slowdown is detected when steps get slower."""
        timer = StepTimer(window_size=20)  # Smaller window for test
        detector = SlowdownDetector(
            threshold=1.5,
            min_steps_for_baseline=15,
            check_interval=3,
        )
        
        # First 25 steps: fast (2ms)
        for step in range(25):
            timer.start_step(step)
            time.sleep(0.002)
            timer.end_step()
            detector.check(timer)
        
        # Next 25 steps: slow (15ms) - should trigger slowdown
        detected_slowdown = False
        for step in range(25, 50):
            timer.start_step(step)
            time.sleep(0.015)  # 7.5x slower
            timer.end_step()
            
            event = detector.check(timer)
            if event is not None:
                detected_slowdown = True
                assert event.slowdown_factor >= 1.5
                assert event.likely_cause is not None
                assert event.suggestion is not None
        
        assert detected_slowdown, "Should have detected slowdown"
        assert detector.has_slowdown
        
    def test_memory_correlated_diagnosis(self):
        """Test that memory growth is correlated with slowdown."""
        timer = StepTimer()
        detector = SlowdownDetector(
            threshold=1.5,
            min_steps_for_baseline=20,
            check_interval=5,
        )
        
        # First 30 steps: fast, low memory
        for step in range(30):
            timer.start_step(step)
            time.sleep(0.005)
            timer.end_step(memory_mb=1000.0)
            detector.check(timer)
        
        # Next 30 steps: slow, high memory (leak simulation)
        for step in range(30, 60):
            timer.start_step(step)
            time.sleep(0.02)
            timer.end_step(memory_mb=3000.0 + step * 50)  # Growing memory
            
            event = detector.check(timer)
            if event is not None and event.memory_growth_mb:
                # Should mention memory in diagnosis
                assert "memory" in event.likely_cause.lower() or event.memory_growth_mb > 500


class TestThroughputTracker:
    """Tests for ThroughputTracker."""
    
    def test_basic_throughput(self):
        """Test basic throughput tracking."""
        tracker = ThroughputTracker()
        
        for _ in range(10):
            tracker.start_step()
            time.sleep(0.01)  # 10ms per step
            tracker.end_step(num_tokens=1000, num_samples=8)
        
        report = tracker.report()
        
        assert report.total_tokens == 10000
        assert report.total_samples == 80
        assert report.mean_tokens_per_sec is not None
        # ~100k tokens/sec (1000 tokens in 10ms)
        assert report.mean_tokens_per_sec > 50000
        
    def test_batch_size_calculation(self):
        """Test throughput with batch size and seq length."""
        tracker = ThroughputTracker()
        
        tracker.start_step()
        time.sleep(0.01)
        tracker.end_step(batch_size=8, seq_length=512)
        
        report = tracker.report()
        assert report.total_tokens == 8 * 512
        
    def test_recent_throughput(self):
        """Test recent throughput calculation."""
        tracker = ThroughputTracker(window_size=5)
        
        for _ in range(10):
            tracker.start_step()
            time.sleep(0.005)
            tracker.end_step(num_tokens=100)
        
        recent = tracker.get_recent_throughput(window=5)
        assert recent["tokens_per_sec"] is not None
        assert recent["tokens_per_sec"] > 0


class TestGPUProfiler:
    """Tests for GPUProfiler (mocked - no real GPU required)."""
    
    def test_unavailable_graceful_degradation(self):
        """Test that profiler works when CUDA unavailable."""
        profiler = GPUProfiler()
        
        # Should not crash even without GPU
        snapshot = profiler.record_step(0)
        # snapshot may be None if no GPU
        
        report = profiler.report()
        assert report is not None
        # Should have some default/empty values
        
    def test_report_structure(self):
        """Test that report has correct structure."""
        profiler = GPUProfiler()
        report = profiler.report()
        
        assert hasattr(report, "peak_memory_mb")
        assert hasattr(report, "final_memory_mb")
        assert hasattr(report, "memory_growth_mb")
        assert hasattr(report, "avg_fragmentation")
        assert hasattr(report, "memory_pressure")
        assert hasattr(report, "recommendations")


class TestGILMonitor:
    """Tests for GIL contention monitoring."""
    
    def test_basic_monitoring(self):
        """Test basic GIL monitoring."""
        monitor = GILMonitor(sample_interval=0.05)
        monitor.start()
        
        # Do some work
        time.sleep(0.2)
        
        monitor.stop()
        result = monitor.analyze()
        
        assert result is not None
        assert 0 <= result.contention_ratio <= 1
        assert result.total_measured_time_sec > 0
        
    def test_operation_tracking(self):
        """Test tracking specific operations."""
        monitor = GILMonitor()
        
        with monitor.track_operation("test_op"):
            time.sleep(0.02)
        
        with monitor.track_operation("test_op"):
            time.sleep(0.02)
        
        result = monitor.analyze()
        # Should have tracked the operation
        assert len(monitor._operation_times.get("test_op", [])) == 2


class TestDataloaderGILProfiler:
    """Tests for DataloaderGILProfiler."""
    
    def test_batch_tracking(self):
        """Test batch time tracking."""
        profiler = DataloaderGILProfiler()
        
        for _ in range(5):
            time.sleep(0.01)  # Simulate dataloader time
            with profiler.track_batch():
                time.sleep(0.02)  # Simulate compute time
        
        report = profiler.report()
        
        assert "avg_batch_time_ms" in report
        assert "avg_dataloader_time_ms" in report
        assert "dataloader_ratio" in report
        assert report["total_batches"] == 5
        
    def test_recommendations(self):
        """Test that recommendations are generated."""
        profiler = DataloaderGILProfiler()
        
        # Simulate I/O bound scenario (long dataloader time)
        for _ in range(5):
            time.sleep(0.05)  # Long dataloader time
            with profiler.track_batch():
                time.sleep(0.01)  # Short compute time
        
        report = profiler.report()
        assert "recommendation" in report


class TestIntegration:
    """Integration tests for profiling components."""
    
    def test_full_profiling_workflow(self):
        """Test all profiling components together."""
        timer = StepTimer()
        detector = SlowdownDetector(min_steps_for_baseline=10, check_interval=5)
        throughput = ThroughputTracker()
        gpu = GPUProfiler()
        
        for step in range(30):
            # Start step
            timer.start_step(step)
            throughput.start_step()
            
            # Simulate work
            time.sleep(0.005)
            
            # End step
            timer.end_step(memory_mb=1000.0)
            throughput.end_step(num_tokens=1000)
            gpu.record_step(step)
            
            # Check for issues
            detector.check(timer)
        
        # Get summaries
        timer_summary = timer.summary()
        throughput_report = throughput.report()
        gpu_report = gpu.report()
        detector_summary = detector.summary()
        
        # Verify all components produced output
        assert timer_summary["total_steps"] == 30
        assert throughput_report.total_tokens == 30000
        assert gpu_report is not None
        assert "slowdown_detected" in detector_summary
