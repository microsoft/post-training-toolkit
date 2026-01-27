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
    
    def test_trainer_class_map_completeness(self):
        expected_trainers = ["DPO", "PPO", "SFT", "ORPO", "KTO", "CPO", "GRPO"]
        for trainer in expected_trainers:
            assert f"{trainer}Trainer" in TRAINER_CLASS_MAP, f"{trainer}Trainer not in map"
            assert f"{trainer}Config" in TRAINER_CLASS_MAP, f"{trainer}Config not in map"
    
    def test_dpo_trainer_detection(self):
        callback = DiagnosticsCallback()
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "DPOTrainer"
        
        trainer_type = callback._detect_trainer_type(trainer=mock_trainer)
        assert trainer_type == TrainerType.DPO
    
    def test_ppo_trainer_detection(self):
        callback = DiagnosticsCallback()
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "PPOTrainer"
        
        trainer_type = callback._detect_trainer_type(trainer=mock_trainer)
        assert trainer_type == TrainerType.PPO
    
    def test_grpo_trainer_detection(self):
        callback = DiagnosticsCallback()
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "GRPOTrainer"
        
        trainer_type = callback._detect_trainer_type(trainer=mock_trainer)
        assert trainer_type == TrainerType.GRPO
    
    def test_sft_trainer_detection(self):
        callback = DiagnosticsCallback()
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "SFTTrainer"
        
        trainer_type = callback._detect_trainer_type(trainer=mock_trainer)
        assert trainer_type == TrainerType.SFT
    
    def test_detection_from_config(self):
        callback = DiagnosticsCallback()
        mock_args = MagicMock()
        mock_args.__class__.__name__ = "ORPOConfig"
        
        trainer_type = callback._detect_trainer_type(args=mock_args)
        assert trainer_type == TrainerType.ORPO
    
    def test_detection_from_value_head(self):
        callback = DiagnosticsCallback()
        mock_model = MagicMock()
        mock_model.v_head = MagicMock()
        
        trainer_type = callback._detect_trainer_type(model=mock_model)
        assert trainer_type == TrainerType.PPO
    
    def test_unknown_trainer_fallback(self):
        callback = DiagnosticsCallback()
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "SomeCustomTrainer"
        
        trainer_type = callback._detect_trainer_type(trainer=mock_trainer)
        assert trainer_type == TrainerType.UNKNOWN

class TestMetricMappings:
    
    def test_dpo_metrics_included(self):
        callback = DiagnosticsCallback()
        mappings = callback._build_metric_mappings(TrainerType.DPO)
        
        assert "dpo_loss" in mappings
        assert "win_rate" in mappings
        assert "logps_chosen" in mappings
        assert "logps_rejected" in mappings
        assert "reward_margin" in mappings
    
    def test_ppo_metrics_included(self):
        callback = DiagnosticsCallback()
        mappings = callback._build_metric_mappings(TrainerType.PPO)
        
        assert "ppo_loss" in mappings
        assert "policy_loss" in mappings
        assert "value_loss" in mappings
        assert "entropy" in mappings
        assert "clip_fraction" in mappings
    
    def test_grpo_metrics_included(self):
        callback = DiagnosticsCallback()
        mappings = callback._build_metric_mappings(TrainerType.GRPO)
        
        assert "grpo_loss" in mappings
        assert "group_reward_mean" in mappings
        assert "group_reward_std" in mappings
    
    def test_base_metrics_always_included(self):
        callback = DiagnosticsCallback()
        
        for trainer_type in [TrainerType.DPO, TrainerType.PPO, TrainerType.SFT, 
                            TrainerType.ORPO, TrainerType.KTO, TrainerType.CPO,
                            TrainerType.GRPO]:
            mappings = callback._build_metric_mappings(trainer_type)
            assert "reward_mean" in mappings
            assert "kl" in mappings
            assert "refusal_rate" in mappings

class TestCallbackInitialization:
    
    def test_default_initialization(self):
        callback = DiagnosticsCallback()
        
        assert callback.log_every_n_steps == 1
        assert callback.include_slices == True
        assert callback.verbose == False
        assert callback.enable_snapshots == True
        assert callback.enable_postmortem == True
        assert callback.stop_on_critical == False
    
    def test_custom_initialization(self):
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
        callback = DiagnosticsCallback()
        assert callback.enable_snapshots == True
        assert callback.enable_auto_diff == True
    
    def test_stop_on_critical_disabled_by_default(self):
        callback = DiagnosticsCallback()
        assert callback.stop_on_critical == False
    
    def test_legacy_log_path_compatibility(self):
        import warnings as warn_module
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "metrics.jsonl"
            with warn_module.catch_warnings(record=True) as w:
                warn_module.simplefilter("always")
                callback = DiagnosticsCallback(log_path=log_path)
                assert len(w) == 1
                assert issubclass(w[-1].category, DeprecationWarning)
            
            assert callback.run_dir == log_path.parent

class TestCallbackLifecycle:
    
    def test_on_train_begin_creates_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = DiagnosticsCallback(run_dir=tmpdir, enable_snapshots=False)
            
            mock_args = MagicMock()
            mock_args.__class__.__name__ = "DPOConfig"
            mock_args.to_dict.return_value = {"learning_rate": 1e-5, "beta": 0.1}
            
            mock_state = MagicMock()
            mock_state.global_step = 0
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
            
            assert (Path(tmpdir) / "run_metadata_start.json").exists()
            assert (Path(tmpdir) / "run_metadata.json").exists()
            assert (Path(tmpdir) / "metrics.jsonl").exists()
    
    def test_on_train_begin_captures_provenance(self):
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
            
            with open(Path(tmpdir) / "run_metadata_start.json") as f:
                metadata = json.load(f)
            
            assert metadata["trainer_type"] == "dpo"
            assert metadata["model_name"] == "meta-llama/Llama-2-7b"
            assert "hardware" in metadata
            assert "package_versions" in metadata
            assert "config_hash" in metadata
    
    def test_find_metric(self):
        callback = DiagnosticsCallback()
        callback._metric_mappings = callback._build_metric_mappings(TrainerType.DPO)
        
        raw_logs = {
            "loss": 0.55,
            "train_loss": 0.55,
            "rewards/accuracies": 0.62,
            "logps/chosen": -2.5,
            "logps/rejected": -3.1,
        }
        
        assert callback._find_metric(raw_logs, "dpo_loss") == 0.55
        assert callback._find_metric(raw_logs, "win_rate") == 0.62
        assert callback._find_metric(raw_logs, "logps_chosen") == -2.5
        assert callback._find_metric(raw_logs, "logps_rejected") == -3.1
    
    def test_critical_failure_detection(self):
        callback = DiagnosticsCallback()
        
        metrics_nan = {"loss": float("nan"), "reward": 0.5}
        result = callback._check_critical_failure(metrics_nan)
        assert result is not None
        assert "NaN" in result
        
        metrics_inf = {"loss": float("inf"), "reward": 0.5}
        result = callback._check_critical_failure(metrics_inf)
        assert result is not None
        assert "Inf" in result
        
        metrics_ok = {"loss": 0.5, "reward": 0.5}
        result = callback._check_critical_failure(metrics_ok)
        assert result is None
    
    def test_on_train_end_writes_final_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = DiagnosticsCallback(run_dir=tmpdir, enable_snapshots=False)
            
            mock_args = MagicMock()
            mock_args.__class__.__name__ = "DPOConfig"
            mock_args.to_dict.return_value = {}
            mock_state = MagicMock()
            mock_state.global_step = 0
            mock_control = MagicMock()
            
            mock_model = MagicMock(spec=[])
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
            
            mock_state.global_step = 100
            callback.on_train_end(mock_args, mock_state, mock_control)
            
            assert (Path(tmpdir) / "run_metadata_final.json").exists()
            
            with open(Path(tmpdir) / "run_metadata_final.json") as f:
                final_metadata = json.load(f)
            
            assert final_metadata["status"] == "completed"
            assert final_metadata["total_steps"] == 100

class TestResumeValidation:
    
    def test_resume_validation_runs_on_nonzero_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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
            mock_state.global_step = 100
            mock_control = MagicMock()
            
            mock_model = MagicMock(spec=[])
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
            
            assert callback._resume_validation_result is not None
            assert (Path(tmpdir) / "resume_validation.json").exists()

class TestExperimentTracker:
    
    def test_wandb_tracker_initialization(self):
        callback = DiagnosticsCallback(
            experiment_tracker="wandb",
            experiment_name="test_run",
        )
        assert callback._experiment_tracker_type == "wandb"
        assert callback._experiment_name == "test_run"
    
    def test_mlflow_tracker_initialization(self):
        callback = DiagnosticsCallback(
            experiment_tracker="mlflow",
            experiment_name="test_run",
        )
        assert callback._experiment_tracker_type == "mlflow"
    
    def test_tensorboard_tracker_initialization(self):
        callback = DiagnosticsCallback(
            experiment_tracker="tensorboard",
            experiment_name="test_run",
        )
        assert callback._experiment_tracker_type == "tensorboard"
    
    def test_no_tracker_by_default(self):
        callback = DiagnosticsCallback()
        assert callback._experiment_tracker_type is None
