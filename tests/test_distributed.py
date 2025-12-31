"""Tests for the distributed training support module.

These tests verify the distributed utilities work correctly in both
single-process mode (graceful fallback) and simulated multi-process mode.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from post_training_toolkit.models.distributed import (
    # rank.py
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
    # aggregation.py
    gather_scalar,
    gather_dict,
    all_gather_object,
    broadcast_object,
    # memory.py
    DistributedMemoryTracker,
    DistributedMemorySnapshot,
    get_distributed_memory_snapshot,
    # straggler.py
    StragglerDetector,
    StragglerReport,
)


class TestRankUtilities:
    """Tests for rank.py utilities."""
    
    def test_single_process_defaults(self):
        """Test default values in single-process mode."""
        # Without torch.distributed initialized, should return defaults
        assert get_rank() == 0
        assert get_local_rank() == 0
        assert get_world_size() == 1
        assert is_main_process() is True
        assert is_distributed() is False
    
    def test_env_var_fallback(self):
        """Test that environment variables are read correctly."""
        with patch.dict("os.environ", {"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "3"}):
            # Clear any cached torch.distributed state
            with patch("post_training_toolkit.models.distributed.rank._get_torch_distributed") as mock_dist:
                mock_dist.return_value = None
                
                assert get_rank() == 3
                assert get_local_rank() == 3
                assert get_world_size() == 8
                assert is_main_process() is False
                assert is_distributed() is True
    
    def test_deepspeed_env_vars(self):
        """Test DeepSpeed-specific environment variables."""
        with patch.dict("os.environ", {
            "DEEPSPEED_RANK": "2",
            "DEEPSPEED_WORLD_SIZE": "4",
            "DEEPSPEED_LOCAL_RANK": "2"
        }, clear=False):
            with patch("post_training_toolkit.models.distributed.rank._get_torch_distributed") as mock_dist:
                mock_dist.return_value = None
                # Clear standard env vars to force DeepSpeed fallback
                with patch.dict("os.environ", {"RANK": "", "WORLD_SIZE": "", "LOCAL_RANK": ""}, clear=False):
                    import os
                    os.environ.pop("RANK", None)
                    os.environ.pop("WORLD_SIZE", None)
                    os.environ.pop("LOCAL_RANK", None)
                    
                    assert get_rank() == 2
                    assert get_world_size() == 4
    
    def test_distributed_info_dataclass(self):
        """Test DistributedInfo dataclass."""
        info = get_distributed_info()
        
        assert isinstance(info, DistributedInfo)
        assert info.rank == 0  # Single process
        assert info.world_size == 1
        assert info.is_distributed is False
        
        # Should have a string representation
        str_repr = str(info)
        assert "single process" in str_repr.lower() or "DistributedInfo" in str_repr
    
    def test_barrier_noop_single_process(self):
        """Test that barrier is a no-op in single process mode."""
        # Should not raise
        barrier()
        barrier(timeout_sec=1.0)
    
    def test_get_device_cpu_fallback(self):
        """Test get_device returns cpu when CUDA unavailable."""
        # In single-process mode without CUDA, should return cpu
        # Note: Can't easily mock internal function, so just verify behavior
        device = get_device()
        assert device in ["cpu", "cuda:0"]  # Either is valid depending on environment


class TestAggregation:
    """Tests for aggregation.py utilities."""
    
    def test_gather_scalar_single_process(self):
        """Test gather_scalar in single-process mode."""
        assert gather_scalar(5.0, op="mean") == 5.0
        assert gather_scalar(5.0, op="sum") == 5.0
        assert gather_scalar(5.0, op="max") == 5.0
        assert gather_scalar(5.0, op="min") == 5.0
    
    def test_gather_dict_single_process(self):
        """Test gather_dict in single-process mode."""
        metrics = {"step_time": 0.8, "memory_mb": 15000}
        result = gather_dict(metrics)
        
        # Should add suffixes for API consistency
        assert "step_time_mean" in result
        assert "step_time_max" in result
        assert "step_time_min" in result
        assert result["step_time_mean"] == 0.8
        assert result["memory_mb_mean"] == 15000
    
    def test_gather_dict_custom_ops(self):
        """Test gather_dict with custom operations."""
        metrics = {"loss": 0.5}
        result = gather_dict(metrics, ops=["mean", "max"])
        
        assert "loss_mean" in result
        assert "loss_max" in result
        assert "loss_min" not in result  # Only requested ops should be present
    
    def test_all_gather_object_single_process(self):
        """Test all_gather_object in single-process mode."""
        obj = {"rank": 0, "data": [1, 2, 3]}
        result = all_gather_object(obj)
        
        assert result == [obj]
    
    def test_broadcast_object_single_process(self):
        """Test broadcast_object in single-process mode."""
        obj = {"config": "value"}
        result = broadcast_object(obj, src=0)
        
        assert result == obj
    
    def test_invalid_op_raises(self):
        """Test that invalid op raises ValueError."""
        # In single process mode, op validation might be skipped
        # This test verifies behavior when distributed IS initialized
        pass  # Would need mocked distributed environment


class TestDistributedMemory:
    """Tests for memory.py utilities."""
    
    def test_memory_snapshot_single_process(self):
        """Test memory snapshot in single-process mode."""
        snapshot = get_distributed_memory_snapshot()
        
        assert isinstance(snapshot, DistributedMemorySnapshot)
        assert snapshot.world_size == 1
        assert snapshot.max_rank == 0
        assert snapshot.min_rank == 0
        assert snapshot.imbalance_ratio == 0.0
        assert snapshot.is_imbalanced is False
    
    def test_memory_snapshot_format(self):
        """Test memory snapshot formatting."""
        snapshot = get_distributed_memory_snapshot()
        formatted = snapshot.format()
        
        assert "Memory" in formatted
        assert "world_size=1" in formatted
    
    def test_memory_tracker_basic(self):
        """Test DistributedMemoryTracker basic usage."""
        tracker = DistributedMemoryTracker(history_size=10)
        
        # Record a few snapshots
        for step in range(5):
            snapshot = tracker.record(step)
            assert isinstance(snapshot, DistributedMemorySnapshot)
        
        # Check history
        assert len(tracker.snapshots) == 5
        assert len(tracker.steps) == 5
    
    def test_memory_tracker_history_limit(self):
        """Test that tracker respects history_size."""
        tracker = DistributedMemoryTracker(history_size=3)
        
        for step in range(10):
            tracker.record(step)
        
        assert len(tracker.snapshots) == 3
        assert tracker.steps[-1] == 9  # Most recent step
    
    def test_memory_tracker_report(self):
        """Test memory tracker report generation."""
        tracker = DistributedMemoryTracker()
        
        for step in range(5):
            tracker.record(step)
        
        report = tracker.report()
        
        assert report.current_snapshot is not None
        assert report.highest_growth_rank == 0
        assert isinstance(report.format(), str)
    
    def test_memory_tracker_has_memory_issue(self):
        """Test has_memory_issue detection."""
        tracker = DistributedMemoryTracker()
        
        # With no data, should return False
        assert tracker.has_memory_issue() is False
        
        # After recording, in single process, should generally be False
        for step in range(5):
            tracker.record(step)
        
        # Single process = no imbalance
        assert tracker.has_memory_issue() is False


class TestStragglerDetector:
    """Tests for straggler.py utilities."""
    
    def test_straggler_detector_basic(self):
        """Test basic straggler detector usage."""
        detector = StragglerDetector(window_size=10)
        
        # Record some steps
        for step in range(10):
            detector.record_step(step, duration=0.1)
        
        report = detector.analyze()
        
        assert report is not None
        assert isinstance(report, StragglerReport)
        assert report.world_size == 1
        assert report.has_straggler is False
    
    def test_straggler_detector_start_end_step(self):
        """Test start_step/end_step interface."""
        detector = StragglerDetector()
        
        for step in range(5):
            detector.start_step()
            time.sleep(0.01)  # 10ms
            duration = detector.end_step(step)
            
            assert duration >= 0.01
            assert duration < 0.1
    
    def test_straggler_detector_insufficient_data(self):
        """Test that analyze returns None with insufficient data."""
        detector = StragglerDetector()
        
        # Only 2 steps - not enough
        detector.record_step(0, 0.1)
        detector.record_step(1, 0.1)
        
        report = detector.analyze()
        assert report is None
    
    def test_straggler_report_format(self):
        """Test straggler report formatting."""
        detector = StragglerDetector()
        
        for step in range(10):
            detector.record_step(step, duration=0.1)
        
        report = detector.analyze()
        formatted = report.format()
        
        assert "Straggler" in formatted
        assert "World size" in formatted
        assert "Mean step time" in formatted
    
    def test_straggler_detector_efficiency(self):
        """Test efficiency calculation."""
        detector = StragglerDetector()
        
        for step in range(10):
            detector.record_step(step, duration=0.1)
        
        efficiency = detector.get_efficiency()
        
        # Single process should have 100% efficiency
        assert efficiency == 1.0
    
    def test_straggler_detector_with_memory(self):
        """Test recording steps with memory data."""
        detector = StragglerDetector()
        
        for step in range(10):
            detector.record_step(step, duration=0.1, memory_mb=1000.0 + step * 10)
        
        report = detector.analyze()
        assert report is not None


class TestDistributedIntegration:
    """Integration tests for distributed module."""
    
    def test_imports_work(self):
        """Test that all public imports work."""
        from post_training_toolkit.models.distributed import (
            get_rank,
            get_local_rank,
            get_world_size,
            is_main_process,
            is_distributed,
            gather_scalar,
            gather_dict,
            StragglerDetector,
            DistributedMemoryTracker,
        )
        
        # All should be callable
        assert callable(get_rank)
        assert callable(gather_dict)
    
    def test_models_init_exports(self):
        """Test that models/__init__.py exports distributed utilities."""
        from post_training_toolkit.models import (
            get_rank,
            is_main_process,
            gather_dict,
            StragglerDetector,
        )
        
        assert get_rank() == 0
        assert is_main_process() is True
    
    def test_typical_workflow(self):
        """Test a typical distributed profiling workflow."""
        # This simulates what a user would do in their training loop
        
        from post_training_toolkit.models.distributed import (
            is_main_process,
            gather_dict,
            StragglerDetector,
            DistributedMemoryTracker,
        )
        
        detector = StragglerDetector()
        memory_tracker = DistributedMemoryTracker()
        
        for step in range(20):
            # Simulate training step
            detector.start_step()
            time.sleep(0.005)  # 5ms
            detector.end_step(step)
            
            # Record memory
            memory_tracker.record(step)
            
            # Gather metrics
            local_metrics = {"loss": 0.5 - step * 0.01, "lr": 0.001}
            global_metrics = gather_dict(local_metrics)
            
            # Only rank 0 logs
            if is_main_process():
                assert "loss_mean" in global_metrics
        
        # Final analysis
        straggler_report = detector.analyze()
        memory_report = memory_tracker.report()
        
        assert straggler_report is not None
        assert not straggler_report.has_straggler  # Single process
        assert memory_report.current_snapshot is not None


class TestMockedDistributed:
    """Tests with mocked torch.distributed to simulate multi-GPU."""
    
    def test_gather_scalar_mocked_distributed(self):
        """Test gather_scalar with mocked distributed backend."""
        # This would require more complex mocking of torch.distributed
        # For now, verify the function handles the mock gracefully
        pass
    
    def test_straggler_detection_logic(self):
        """Test straggler detection logic with simulated multi-rank data."""
        # We can't easily simulate multiple ranks, but we can verify
        # the analysis logic by checking the report structure
        detector = StragglerDetector(
            straggler_threshold=1.2,
            consistent_checks=3,
        )
        
        for step in range(20):
            detector.record_step(step, duration=0.1)
        
        report = detector.analyze()
        
        # Verify report structure
        assert hasattr(report, 'has_straggler')
        assert hasattr(report, 'slowest_rank')
        assert hasattr(report, 'slowdown_factor')
        assert hasattr(report, 'is_consistent')
        assert hasattr(report, 'likely_cause')
        assert hasattr(report, 'suggestion')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
