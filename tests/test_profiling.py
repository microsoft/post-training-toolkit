
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
    
    def test_basic_timing(self):
        timer = StepTimer()
        
        timer.start_step(0)
        time.sleep(0.01)
        timing = timer.end_step()
        
        assert timing is not None
        assert timing.step == 0
        assert timing.duration_sec >= 0.01
        assert timing.duration_sec < 0.1
        
    def test_timing_accumulation(self):
        timer = StepTimer()
        
        for step in range(10):
            timer.start_step(step)
            time.sleep(0.005)
            timer.end_step()
        
        assert timer.total_steps == 10
        assert timer.total_time_sec >= 0.05
        
    def test_memory_tracking(self):
        timer = StepTimer()
        
        timer.start_step(0)
        timing = timer.end_step(memory_mb=1000.0)
        
        assert timing.memory_mb == 1000.0
        
    def test_baseline_calculation(self):
        timer = StepTimer(window_size=10)
        
        for step in range(30):
            timer.start_step(step)
            time.sleep(0.005)
            timer.end_step()
        
        baseline = timer.get_baseline_duration()
        assert baseline is not None
        assert baseline >= 0.004
        assert baseline < 0.02
        
    def test_summary(self):
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
    
    def test_no_slowdown(self):
        timer = StepTimer()
        detector = SlowdownDetector(
            threshold=1.5,
            min_steps_for_baseline=10,
            check_interval=5,
        )
        
        for step in range(50):
            timer.start_step(step)
            time.sleep(0.005)
            timer.end_step()
            
            event = detector.check(timer)
            if step > 20:
                assert event is None or event.slowdown_factor < 1.5
                
    def test_detects_slowdown(self):
        timer = StepTimer(window_size=20)
        detector = SlowdownDetector(
            threshold=1.5,
            min_steps_for_baseline=15,
            check_interval=3,
        )
        
        for step in range(25):
            timer.start_step(step)
            time.sleep(0.002)
            timer.end_step()
            detector.check(timer)
        
        detected_slowdown = False
        for step in range(25, 50):
            timer.start_step(step)
            time.sleep(0.015)
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
        timer = StepTimer()
        detector = SlowdownDetector(
            threshold=1.5,
            min_steps_for_baseline=20,
            check_interval=5,
        )
        
        for step in range(30):
            timer.start_step(step)
            time.sleep(0.005)
            timer.end_step(memory_mb=1000.0)
            detector.check(timer)
        
        for step in range(30, 60):
            timer.start_step(step)
            time.sleep(0.02)
            timer.end_step(memory_mb=3000.0 + step * 50)
            
            event = detector.check(timer)
            if event is not None and event.memory_growth_mb:
                assert "memory" in event.likely_cause.lower() or event.memory_growth_mb > 500

class TestThroughputTracker:
    
    def test_basic_throughput(self):
        tracker = ThroughputTracker()
        
        for _ in range(10):
            tracker.start_step()
            time.sleep(0.01)
            tracker.end_step(num_tokens=1000, num_samples=8)
        
        report = tracker.report()
        
        assert report.total_tokens == 10000
        assert report.total_samples == 80
        assert report.mean_tokens_per_sec is not None
        assert report.mean_tokens_per_sec > 50000
        
    def test_batch_size_calculation(self):
        tracker = ThroughputTracker()
        
        tracker.start_step()
        time.sleep(0.01)
        tracker.end_step(batch_size=8, seq_length=512)
        
        report = tracker.report()
        assert report.total_tokens == 8 * 512
        
    def test_recent_throughput(self):
        tracker = ThroughputTracker(window_size=5)
        
        for _ in range(10):
            tracker.start_step()
            time.sleep(0.005)
            tracker.end_step(num_tokens=100)
        
        recent = tracker.get_recent_throughput(window=5)
        assert recent["tokens_per_sec"] is not None
        assert recent["tokens_per_sec"] > 0

class TestGPUProfiler:
    
    def test_unavailable_graceful_degradation(self):
        profiler = GPUProfiler()
        
        snapshot = profiler.record_step(0)
        
        report = profiler.report()
        assert report is not None
        
    def test_report_structure(self):
        profiler = GPUProfiler()
        report = profiler.report()
        
        assert hasattr(report, "peak_memory_mb")
        assert hasattr(report, "final_memory_mb")
        assert hasattr(report, "memory_growth_mb")
        assert hasattr(report, "avg_fragmentation")
        assert hasattr(report, "memory_pressure")
        assert hasattr(report, "recommendations")

class TestGILMonitor:
    
    def test_basic_monitoring(self):
        monitor = GILMonitor(sample_interval=0.05)
        monitor.start()
        
        time.sleep(0.2)
        
        monitor.stop()
        result = monitor.analyze()
        
        assert result is not None
        assert 0 <= result.contention_ratio <= 1
        assert result.total_measured_time_sec > 0
        
    def test_operation_tracking(self):
        monitor = GILMonitor()
        
        with monitor.track_operation("test_op"):
            time.sleep(0.02)
        
        with monitor.track_operation("test_op"):
            time.sleep(0.02)
        
        result = monitor.analyze()
        assert len(monitor._operation_times.get("test_op", [])) == 2

class TestDataloaderGILProfiler:
    
    def test_batch_tracking(self):
        profiler = DataloaderGILProfiler()
        
        for _ in range(5):
            time.sleep(0.01)
            with profiler.track_batch():
                time.sleep(0.02)
        
        report = profiler.report()
        
        assert "avg_batch_time_ms" in report
        assert "avg_dataloader_time_ms" in report
        assert "dataloader_ratio" in report
        assert report["total_batches"] == 5
        
    def test_recommendations(self):
        profiler = DataloaderGILProfiler()
        
        for _ in range(5):
            time.sleep(0.05)
            with profiler.track_batch():
                time.sleep(0.01)
        
        report = profiler.report()
        assert "recommendation" in report

class TestIntegration:
    
    def test_full_profiling_workflow(self):
        timer = StepTimer()
        detector = SlowdownDetector(min_steps_for_baseline=10, check_interval=5)
        throughput = ThroughputTracker()
        gpu = GPUProfiler()
        
        for step in range(30):
            timer.start_step(step)
            throughput.start_step()
            
            time.sleep(0.005)
            
            timer.end_step(memory_mb=1000.0)
            throughput.end_step(num_tokens=1000)
            gpu.record_step(step)
            
            detector.check(timer)
        
        timer_summary = timer.summary()
        throughput_report = throughput.report()
        gpu_report = gpu.report()
        detector_summary = detector.summary()
        
        assert timer_summary["total_steps"] == 30
        assert throughput_report.total_tokens == 30000
        assert gpu_report is not None
        assert "slowdown_detected" in detector_summary
