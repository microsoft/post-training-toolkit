
import pytest
import time
from unittest.mock import patch, MagicMock

from post_training_toolkit.models.distributed import (
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
    gather_scalar,
    gather_dict,
    all_gather_object,
    broadcast_object,
    DistributedMemoryTracker,
    DistributedMemorySnapshot,
    get_distributed_memory_snapshot,
    StragglerDetector,
    StragglerReport,
)

class TestRankUtilities:
    
    def test_single_process_defaults(self):
        assert get_rank() == 0
        assert get_local_rank() == 0
        assert get_world_size() == 1
        assert is_main_process() is True
        assert is_distributed() is False
    
    def test_env_var_fallback(self):
        with patch.dict("os.environ", {"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "3"}):
            with patch("post_training_toolkit.models.distributed.rank._get_torch_distributed") as mock_dist:
                mock_dist.return_value = None
                
                assert get_rank() == 3
                assert get_local_rank() == 3
                assert get_world_size() == 8
                assert is_main_process() is False
                assert is_distributed() is True
    
    def test_deepspeed_env_vars(self):
        with patch.dict("os.environ", {
            "DEEPSPEED_RANK": "2",
            "DEEPSPEED_WORLD_SIZE": "4",
            "DEEPSPEED_LOCAL_RANK": "2"
        }, clear=False):
            with patch("post_training_toolkit.models.distributed.rank._get_torch_distributed") as mock_dist:
                mock_dist.return_value = None
                with patch.dict("os.environ", {"RANK": "", "WORLD_SIZE": "", "LOCAL_RANK": ""}, clear=False):
                    import os
                    os.environ.pop("RANK", None)
                    os.environ.pop("WORLD_SIZE", None)
                    os.environ.pop("LOCAL_RANK", None)
                    
                    assert get_rank() == 2
                    assert get_world_size() == 4
    
    def test_distributed_info_dataclass(self):
        info = get_distributed_info()
        
        assert isinstance(info, DistributedInfo)
        assert info.rank == 0
        assert info.world_size == 1
        assert info.is_distributed is False
        
        str_repr = str(info)
        assert "single process" in str_repr.lower() or "DistributedInfo" in str_repr
    
    def test_barrier_noop_single_process(self):
        barrier()
        barrier(timeout_sec=1.0)
    
    def test_get_device_cpu_fallback(self):
        device = get_device()
        assert device in ["cpu", "cuda:0"]

class TestAggregation:
    
    def test_gather_scalar_single_process(self):
        assert gather_scalar(5.0, op="mean") == 5.0
        assert gather_scalar(5.0, op="sum") == 5.0
        assert gather_scalar(5.0, op="max") == 5.0
        assert gather_scalar(5.0, op="min") == 5.0
    
    def test_gather_dict_single_process(self):
        metrics = {"step_time": 0.8, "memory_mb": 15000}
        result = gather_dict(metrics)
        
        assert "step_time_mean" in result
        assert "step_time_max" in result
        assert "step_time_min" in result
        assert result["step_time_mean"] == 0.8
        assert result["memory_mb_mean"] == 15000
    
    def test_gather_dict_custom_ops(self):
        metrics = {"loss": 0.5}
        result = gather_dict(metrics, ops=["mean", "max"])
        
        assert "loss_mean" in result
        assert "loss_max" in result
        assert "loss_min" not in result
    
    def test_all_gather_object_single_process(self):
        obj = {"rank": 0, "data": [1, 2, 3]}
        result = all_gather_object(obj)
        
        assert result == [obj]
    
    def test_broadcast_object_single_process(self):
        obj = {"config": "value"}
        result = broadcast_object(obj, src=0)
        
        assert result == obj
    
    def test_invalid_op_raises(self):
        pass

class TestDistributedMemory:
    
    def test_memory_snapshot_single_process(self):
        snapshot = get_distributed_memory_snapshot()
        
        assert isinstance(snapshot, DistributedMemorySnapshot)
        assert snapshot.world_size == 1
        assert snapshot.max_rank == 0
        assert snapshot.min_rank == 0
        assert snapshot.imbalance_ratio == 0.0
        assert snapshot.is_imbalanced is False
    
    def test_memory_snapshot_format(self):
        snapshot = get_distributed_memory_snapshot()
        formatted = snapshot.format()
        
        assert "Memory" in formatted
        assert "world_size=1" in formatted
    
    def test_memory_tracker_basic(self):
        tracker = DistributedMemoryTracker(history_size=10)
        
        for step in range(5):
            snapshot = tracker.record(step)
            assert isinstance(snapshot, DistributedMemorySnapshot)
        
        assert len(tracker.snapshots) == 5
        assert len(tracker.steps) == 5
    
    def test_memory_tracker_history_limit(self):
        tracker = DistributedMemoryTracker(history_size=3)
        
        for step in range(10):
            tracker.record(step)
        
        assert len(tracker.snapshots) == 3
        assert tracker.steps[-1] == 9
    
    def test_memory_tracker_report(self):
        tracker = DistributedMemoryTracker()
        
        for step in range(5):
            tracker.record(step)
        
        report = tracker.report()
        
        assert report.current_snapshot is not None
        assert report.highest_growth_rank == 0
        assert isinstance(report.format(), str)
    
    def test_memory_tracker_has_memory_issue(self):
        tracker = DistributedMemoryTracker()
        
        assert tracker.has_memory_issue() is False
        
        for step in range(5):
            tracker.record(step)
        
        assert tracker.has_memory_issue() is False

class TestStragglerDetector:
    
    def test_straggler_detector_basic(self):
        detector = StragglerDetector(window_size=10)
        
        for step in range(10):
            detector.record_step(step, duration=0.1)
        
        report = detector.analyze()
        
        assert report is not None
        assert isinstance(report, StragglerReport)
        assert report.world_size == 1
        assert report.has_straggler is False
    
    def test_straggler_detector_start_end_step(self):
        detector = StragglerDetector()
        
        for step in range(5):
            detector.start_step()
            time.sleep(0.01)
            duration = detector.end_step(step)
            
            assert duration >= 0.01
            assert duration < 0.1
    
    def test_straggler_detector_insufficient_data(self):
        detector = StragglerDetector()
        
        detector.record_step(0, 0.1)
        detector.record_step(1, 0.1)
        
        report = detector.analyze()
        assert report is None
    
    def test_straggler_report_format(self):
        detector = StragglerDetector()
        
        for step in range(10):
            detector.record_step(step, duration=0.1)
        
        report = detector.analyze()
        formatted = report.format()
        
        assert "Straggler" in formatted
        assert "World size" in formatted
        assert "Mean step time" in formatted
    
    def test_straggler_detector_efficiency(self):
        detector = StragglerDetector()
        
        for step in range(10):
            detector.record_step(step, duration=0.1)
        
        efficiency = detector.get_efficiency()
        
        assert efficiency == 1.0
    
    def test_straggler_detector_with_memory(self):
        detector = StragglerDetector()
        
        for step in range(10):
            detector.record_step(step, duration=0.1, memory_mb=1000.0 + step * 10)
        
        report = detector.analyze()
        assert report is not None

class TestDistributedIntegration:
    
    def test_imports_work(self):
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
        
        assert callable(get_rank)
        assert callable(gather_dict)
    
    def test_models_init_exports(self):
        from post_training_toolkit.models import (
            get_rank,
            is_main_process,
            gather_dict,
            StragglerDetector,
        )
        
        assert get_rank() == 0
        assert is_main_process() is True
    
    def test_typical_workflow(self):
        
        from post_training_toolkit.models.distributed import (
            is_main_process,
            gather_dict,
            StragglerDetector,
            DistributedMemoryTracker,
        )
        
        detector = StragglerDetector()
        memory_tracker = DistributedMemoryTracker()
        
        for step in range(20):
            detector.start_step()
            time.sleep(0.005)
            detector.end_step(step)
            
            memory_tracker.record(step)
            
            local_metrics = {"loss": 0.5 - step * 0.01, "lr": 0.001}
            global_metrics = gather_dict(local_metrics)
            
            if is_main_process():
                assert "loss_mean" in global_metrics
        
        straggler_report = detector.analyze()
        memory_report = memory_tracker.report()
        
        assert straggler_report is not None
        assert not straggler_report.has_straggler
        assert memory_report.current_snapshot is not None

class TestMockedDistributed:
    
    def test_gather_scalar_mocked_distributed(self):
        pass
    
    def test_straggler_detection_logic(self):
        detector = StragglerDetector(
            straggler_threshold=1.2,
            consistent_checks=3,
        )
        
        for step in range(20):
            detector.record_step(step, duration=0.1)
        
        report = detector.analyze()
        
        assert hasattr(report, 'has_straggler')
        assert hasattr(report, 'slowest_rank')
        assert hasattr(report, 'slowdown_factor')
        assert hasattr(report, 'is_consistent')
        assert hasattr(report, 'likely_cause')
        assert hasattr(report, 'suggestion')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
