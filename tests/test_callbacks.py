"""Smoke tests for TRL integration."""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from post_training_toolkit.integrations.trl import (
    DiagnosticsCallback,
    TrainerType,
    TRAINER_CLASS_MAP,
)


class TestTrainerTypeDetection:
    """Tests for trainer type auto-detection."""
    
    def test_trainer_class_map_completeness(self):
        """Ensure all expected trainers are mapped."""
        expected_trainers = ["DPO", "PPO", "SFT", "ORPO", "KTO", "CPO", "GRPO"]
        for trainer in expected_trainers:
            assert f"{trainer}Trainer" in TRAINER_CLASS_MAP, f"{trainer}Trainer not in map"
            assert f"{trainer}Config" in TRAINER_CLASS_MAP, f"{trainer}Config not in map"
    
    def test_dpo_trainer_detection(self):
        """DPOTrainer should be detected as DPO."""
        callback = DiagnosticsCallback()
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "DPOTrainer"
        
        trainer_type = callback._detect_trainer_type(trainer=mock_trainer)
        assert trainer_type == TrainerType.DPO
    
    def test_ppo_trainer_detection(self):
        """PPOTrainer should be detected as PPO."""
        callback = DiagnosticsCallback()
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "PPOTrainer"
        
        trainer_type = callback._detect_trainer_type(trainer=mock_trainer)
        assert trainer_type == TrainerType.PPO
    
    def test_grpo_trainer_detection(self):
        """GRPOTrainer should be detected as GRPO."""
        callback = DiagnosticsCallback()
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "GRPOTrainer"
        
        trainer_type = callback._detect_trainer_type(trainer=mock_trainer)
        assert trainer_type == TrainerType.GRPO
    
    def test_sft_trainer_detection(self):
        """SFTTrainer should be detected as SFT."""
        callback = DiagnosticsCallback()
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "SFTTrainer"
        
        trainer_type = callback._detect_trainer_type(trainer=mock_trainer)
        assert trainer_type == TrainerType.SFT
    
    def test_detection_from_config(self):
        """Trainer type should be detected from config class name."""
        callback = DiagnosticsCallback()
        mock_args = MagicMock()
        mock_args.__class__.__name__ = "ORPOConfig"
        
        trainer_type = callback._detect_trainer_type(args=mock_args)
        assert trainer_type == TrainerType.ORPO
    
    def test_detection_from_value_head(self):
        """PPO should be detected from model value head."""
        callback = DiagnosticsCallback()
        mock_model = MagicMock()
        mock_model.v_head = MagicMock()  # PPO model has value head
        
        trainer_type = callback._detect_trainer_type(model=mock_model)
        assert trainer_type == TrainerType.PPO
    
    def test_unknown_trainer_fallback(self):
        """Unknown trainer should return UNKNOWN type."""
        callback = DiagnosticsCallback()
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "SomeCustomTrainer"
        
        trainer_type = callback._detect_trainer_type(trainer=mock_trainer)
        assert trainer_type == TrainerType.UNKNOWN


class TestMetricMappings:
    """Tests for metric mapping configuration."""
    
    def test_dpo_metrics_included(self):
        """DPO trainer should have DPO-specific metrics."""
        callback = DiagnosticsCallback()
        mappings = callback._build_metric_mappings(TrainerType.DPO)
        
        assert "dpo_loss" in mappings
        assert "win_rate" in mappings
        assert "logps_chosen" in mappings
        assert "logps_rejected" in mappings
        assert "reward_margin" in mappings
    
    def test_ppo_metrics_included(self):
        """PPO trainer should have PPO-specific metrics."""
        callback = DiagnosticsCallback()
        mappings = callback._build_metric_mappings(TrainerType.PPO)
        
        assert "ppo_loss" in mappings
        assert "policy_loss" in mappings
        assert "value_loss" in mappings
        assert "entropy" in mappings
        assert "clip_fraction" in mappings
    
    def test_grpo_metrics_included(self):
        """GRPO trainer should have GRPO-specific metrics."""
        callback = DiagnosticsCallback()
        mappings = callback._build_metric_mappings(TrainerType.GRPO)
        
        assert "grpo_loss" in mappings
        assert "group_reward_mean" in mappings
        assert "group_reward_std" in mappings
    
    def test_base_metrics_always_included(self):
        """Base metrics should be included for all trainer types."""
        callback = DiagnosticsCallback()
        
        for trainer_type in [TrainerType.DPO, TrainerType.PPO, TrainerType.SFT, 
                            TrainerType.ORPO, TrainerType.KTO, TrainerType.CPO,
                            TrainerType.GRPO]:
            mappings = callback._build_metric_mappings(trainer_type)
            assert "reward_mean" in mappings
            assert "kl" in mappings
            assert "refusal_rate" in mappings


class TestCallbackInitialization:
    """Tests for callback initialization."""
    
    def test_default_initialization(self):
        """Callback should initialize with defaults."""
        callback = DiagnosticsCallback()
        
        assert callback.log_every_n_steps == 1
        assert callback.include_slices == True
        assert callback.verbose == False
        # NOTE: enable_snapshots is now True by default (lab-grade reliability)
        assert callback.enable_snapshots == True
        assert callback.enable_postmortem == True
        # Safe stopping is conservative (off by default)
        assert callback.stop_on_critical == False
    
    def test_custom_initialization(self):
        """Callback should accept custom parameters."""
        callback = DiagnosticsCallback(
            run_dir="my_custom_run",
            log_every_n_steps=5,
            verbose=True,
            enable_snapshots=True,
            snapshot_interval=50,
            enable_auto_diff=True,
            stop_on_critical=True,
            fail_on_resume_mismatch=True,
        )
        
        assert callback.run_dir == Path("my_custom_run")
        assert callback.log_every_n_steps == 5
        assert callback.verbose == True
        assert callback.enable_snapshots == True
        assert callback.snapshot_interval == 50
        assert callback.enable_auto_diff == True
        assert callback.stop_on_critical == True
        assert callback.fail_on_resume_mismatch == True
    
    def test_default_snapshots_enabled(self):
        """Snapshots should be enabled by default."""
        callback = DiagnosticsCallback()
        assert callback.enable_snapshots == True
        assert callback.enable_auto_diff == True
    
    def test_stop_on_critical_disabled_by_default(self):
        """Stop on critical should be disabled by default (conservative)."""
        callback = DiagnosticsCallback()
        assert callback.stop_on_critical == False
    
    def test_legacy_log_path_compatibility(self):
        """Legacy log_path parameter should still work."""
        import warnings as warn_module
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "metrics.jsonl"
            # Should emit deprecation warning
            with warn_module.catch_warnings(record=True) as w:
                warn_module.simplefilter("always")
                callback = DiagnosticsCallback(log_path=log_path)
                assert len(w) == 1
                assert issubclass(w[-1].category, DeprecationWarning)
            
            # run_dir should be derived from log_path
            assert callback.run_dir == log_path.parent


class TestCallbackLifecycle:
    """Tests for callback lifecycle methods."""
    
    def test_on_train_begin_creates_artifacts(self):
        """on_train_begin should create artifact manager and metadata files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = DiagnosticsCallback(run_dir=tmpdir, enable_snapshots=False)
            
            mock_args = MagicMock()
            mock_args.__class__.__name__ = "DPOConfig"
            # Make to_dict return a serializable dict
            mock_args.to_dict.return_value = {"learning_rate": 1e-5, "beta": 0.1}
            
            mock_state = MagicMock()
            mock_state.global_step = 0  # Not resuming
            mock_control = MagicMock()
            mock_trainer = MagicMock()
            mock_trainer.__class__.__name__ = "DPOTrainer"
            mock_trainer.model = MagicMock()
            mock_trainer.model.name_or_path = "test-model"
            mock_trainer.ref_model = None
            mock_trainer.tokenizer = None
            mock_trainer.train_dataset = None
            
            callback.on_train_begin(
                args=mock_args,
                state=mock_state,
                control=mock_control,
                trainer=mock_trainer,
            )
            
            assert callback._initialized == True
            assert callback._trainer_type == TrainerType.DPO
            assert callback._artifact_manager is not None
            
            # Check that metadata files were created
            assert (Path(tmpdir) / "run_metadata_start.json").exists()
            assert (Path(tmpdir) / "run_metadata.json").exists()
            assert (Path(tmpdir) / "metrics.jsonl").exists()
    
    def test_on_train_begin_captures_provenance(self):
        """on_train_begin should capture comprehensive provenance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = DiagnosticsCallback(run_dir=tmpdir, enable_snapshots=False)
            
            mock_args = MagicMock()
            mock_args.__class__.__name__ = "DPOConfig"
            mock_args.to_dict.return_value = {"learning_rate": 1e-5}
            
            mock_state = MagicMock()
            mock_state.global_step = 0
            mock_control = MagicMock()
            mock_trainer = MagicMock()
            mock_trainer.__class__.__name__ = "DPOTrainer"
            mock_trainer.model = MagicMock()
            mock_trainer.model.name_or_path = "meta-llama/Llama-2-7b"
            mock_trainer.ref_model = None
            mock_trainer.tokenizer = MagicMock()
            mock_trainer.tokenizer.name_or_path = "meta-llama/Llama-2-7b"
            mock_trainer.train_dataset = None
            
            callback.on_train_begin(
                args=mock_args,
                state=mock_state,
                control=mock_control,
                trainer=mock_trainer,
            )
            
            # Load and verify metadata
            with open(Path(tmpdir) / "run_metadata_start.json") as f:
                metadata = json.load(f)
            
            assert metadata["trainer_type"] == "dpo"
            assert metadata["model_name"] == "meta-llama/Llama-2-7b"
            assert "hardware" in metadata
            assert "package_versions" in metadata
            assert "config_hash" in metadata
    
    def test_find_metric(self):
        """Metrics should be found correctly from logs."""
        callback = DiagnosticsCallback()
        callback._metric_mappings = callback._build_metric_mappings(TrainerType.DPO)
        
        raw_logs = {
            "loss": 0.55,
            "train_loss": 0.55,
            "rewards/accuracies": 0.62,
            "logps/chosen": -2.5,
            "logps/rejected": -3.1,
        }
        
        # Test _find_metric method
        assert callback._find_metric(raw_logs, "dpo_loss") == 0.55
        assert callback._find_metric(raw_logs, "win_rate") == 0.62
        assert callback._find_metric(raw_logs, "logps_chosen") == -2.5
        assert callback._find_metric(raw_logs, "logps_rejected") == -3.1
    
    def test_critical_failure_detection(self):
        """Critical failures (NaN, Inf) should be detected."""
        callback = DiagnosticsCallback()
        
        # NaN should be detected
        metrics_nan = {"loss": float("nan"), "reward": 0.5}
        result = callback._check_critical_failure(metrics_nan)
        assert result is not None
        assert "NaN" in result
        
        # Inf should be detected
        metrics_inf = {"loss": float("inf"), "reward": 0.5}
        result = callback._check_critical_failure(metrics_inf)
        assert result is not None
        assert "Inf" in result
        
        # Normal metrics should pass
        metrics_ok = {"loss": 0.5, "reward": 0.5}
        result = callback._check_critical_failure(metrics_ok)
        assert result is None
    
    def test_on_train_end_writes_final_metadata(self):
        """on_train_end should write immutable final metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = DiagnosticsCallback(run_dir=tmpdir, enable_snapshots=False)
            
            # Setup - use spec=[] to prevent MagicMock from auto-creating attributes
            mock_args = MagicMock()
            mock_args.__class__.__name__ = "DPOConfig"
            mock_args.to_dict.return_value = {}
            mock_state = MagicMock()
            mock_state.global_step = 0
            mock_control = MagicMock()
            
            # Create model mock that returns serializable values
            mock_model = MagicMock(spec=[])  # Empty spec prevents attr auto-creation
            mock_model.name_or_path = "test-model"
            mock_model.config = MagicMock(spec=[])
            mock_model.config._name_or_path = "test-model"
            mock_model.config._commit_hash = None
            
            mock_trainer = MagicMock()
            mock_trainer.__class__.__name__ = "DPOTrainer"
            mock_trainer.model = mock_model
            mock_trainer.ref_model = None
            mock_trainer.tokenizer = None
            mock_trainer.train_dataset = None
            
            callback.on_train_begin(mock_args, mock_state, mock_control, trainer=mock_trainer)
            
            # Simulate training end
            mock_state.global_step = 100
            callback.on_train_end(mock_args, mock_state, mock_control)
            
            # Check final metadata was written
            assert (Path(tmpdir) / "run_metadata_final.json").exists()
            
            with open(Path(tmpdir) / "run_metadata_final.json") as f:
                final_metadata = json.load(f)
            
            assert final_metadata["status"] == "completed"
            assert final_metadata["total_steps"] == 100


class TestResumeValidation:
    """Tests for resume validation functionality."""
    
    def test_resume_validation_runs_on_nonzero_step(self):
        """Resume validation should run when global_step > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a start metadata file first
            start_meta = {
                "config_hash": "abc123",
                "model_name": "test-model",
                "config": {"training_args": {"learning_rate": 1e-5}},
            }
            (Path(tmpdir) / "run_metadata_start.json").write_text(json.dumps(start_meta))
            
            callback = DiagnosticsCallback(
                run_dir=tmpdir,
                enable_snapshots=False,
                enable_resume_validation=True,
            )
            
            mock_args = MagicMock()
            mock_args.__class__.__name__ = "DPOConfig"
            mock_args.to_dict.return_value = {"learning_rate": 1e-5}
            
            mock_state = MagicMock()
            mock_state.global_step = 100  # Resuming
            mock_control = MagicMock()
            
            # Create model mock that returns serializable values
            mock_model = MagicMock(spec=[])  # Empty spec prevents attr auto-creation
            mock_model.name_or_path = "test-model"
            mock_model.config = MagicMock(spec=[])
            mock_model.config._name_or_path = "test-model"
            mock_model.config._commit_hash = None
            
            mock_trainer = MagicMock()
            mock_trainer.__class__.__name__ = "DPOTrainer"
            mock_trainer.model = mock_model
            mock_trainer.ref_model = None
            mock_trainer.tokenizer = None
            mock_trainer.train_dataset = None
            
            callback.on_train_begin(mock_args, mock_state, mock_control, trainer=mock_trainer)
            
            # Resume validation should have run
            assert callback._resume_validation_result is not None
            assert (Path(tmpdir) / "resume_validation.json").exists()


class TestExperimentTracker:
    """Tests for experiment tracker integration."""
    
    def test_wandb_tracker_initialization(self):
        """WandB tracker should be available but optional."""
        # Without wandb installed, should not crash
        callback = DiagnosticsCallback(
            experiment_tracker="wandb",
            experiment_name="test_run",
        )
        # Tracker is lazy-initialized, so this just sets config
        assert callback._experiment_tracker_type == "wandb"
        assert callback._experiment_name == "test_run"
    
    def test_mlflow_tracker_initialization(self):
        """MLflow tracker should be available but optional."""
        callback = DiagnosticsCallback(
            experiment_tracker="mlflow",
            experiment_name="test_run",
        )
        assert callback._experiment_tracker_type == "mlflow"
    
    def test_tensorboard_tracker_initialization(self):
        """TensorBoard tracker should be available but optional."""
        callback = DiagnosticsCallback(
            experiment_tracker="tensorboard",
            experiment_name="test_run",
        )
        assert callback._experiment_tracker_type == "tensorboard"
    
    def test_no_tracker_by_default(self):
        """No tracker should be enabled by default."""
        callback = DiagnosticsCallback()
        assert callback._experiment_tracker_type is None
