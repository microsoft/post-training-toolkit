
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from post_training_toolkit.models.artifacts import (
    RunArtifactManager,
    RunMetadata,
    compute_config_hash,
    get_git_info,
    get_hardware_info,
    get_package_versions,
    get_model_identity,
    get_tokenizer_identity,
    get_dataset_identity,
    collect_full_provenance,
)

def test_initialize_respects_main_process_override(tmp_path: Path):
    not_main = RunArtifactManager(tmp_path / "nominal", is_main_process_override=False)
    not_main.initialize()
    assert not not_main.metadata_path.exists()

    main_mgr = RunArtifactManager(tmp_path / "main", is_main_process_override=True)
    main_mgr.initialize()
    assert main_mgr.metadata_path.exists()
    assert main_mgr.metadata_start_path.exists()
    assert main_mgr.metrics_path.exists()

def test_initialize_creates_start_metadata(tmp_path: Path):
    mgr = RunArtifactManager(tmp_path, is_main_process_override=True)
    mgr.initialize(
        trainer_type="dpo",
        model_name="test-model",
        config={"learning_rate": 1e-5},
    )
    
    assert mgr.metadata_start_path.exists()
    
    with open(mgr.metadata_start_path) as f:
        metadata = json.load(f)
    
    assert metadata["trainer_type"] == "dpo"
    assert metadata["model_name"] == "test-model"
    assert metadata["status"] == "running"
    assert "hardware" in metadata
    assert "package_versions" in metadata

def test_finalize_creates_final_metadata(tmp_path: Path):
    mgr = RunArtifactManager(tmp_path, is_main_process_override=True)
    mgr.initialize(trainer_type="dpo")
    mgr.finalize(status="completed", total_steps=100)
    
    assert mgr.metadata_final_path.exists()
    
    with open(mgr.metadata_final_path) as f:
        metadata = json.load(f)
    
    assert metadata["status"] == "completed"
    assert metadata["total_steps"] == 100
    assert metadata["end_time"] is not None

class TestConfigHash:
    
    def test_config_hash_is_stable(self):
        config = {"learning_rate": 1e-5, "beta": 0.1, "max_steps": 1000}
        
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        
        assert hash1 == hash2
    
    def test_config_hash_differs_for_different_configs(self):
        config1 = {"learning_rate": 1e-5}
        config2 = {"learning_rate": 1e-4}
        
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        
        assert hash1 != hash2
    
    def test_config_hash_order_independent(self):
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}
        
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        
        assert hash1 == hash2

class TestHardwareInfo:
    
    def test_hardware_info_has_required_fields(self):
        info = get_hardware_info()
        
        assert "hostname" in info
        assert "platform" in info
        assert "cpu_count" in info
        assert "python_version" in info
        assert "cuda_available" in info
        assert "world_size" in info
        assert "global_rank" in info

class TestPackageVersions:
    
    def test_package_versions_includes_core_packages(self):
        versions = get_package_versions()
        
        assert isinstance(versions, dict)
    
    def test_full_package_snapshot(self):
        minimal = get_package_versions(full_env=False)
        full = get_package_versions(full_env=True)
        
        assert len(full) >= len(minimal)

class TestModelIdentity:
    
    def test_model_identity_from_string(self):
        identity = get_model_identity("meta-llama/Llama-2-7b-hf")
        
        assert identity["model_name"] == "meta-llama/Llama-2-7b-hf"
    
    def test_model_identity_from_mock(self):
        mock_model = MagicMock()
        mock_model.name_or_path = "test-model-path"
        
        identity = get_model_identity(mock_model)
        
        assert identity["model_name"] == "test-model-path"
    
    def test_model_identity_from_config(self):
        mock_model = MagicMock(spec=[])
        mock_model.config = MagicMock()
        mock_model.config._name_or_path = "config-model-path"
        mock_model.config._commit_hash = "abc123"
        
        identity = get_model_identity(mock_model)
        
        assert identity["model_name"] == "config-model-path"
        assert identity["model_revision"] == "abc123"

class TestGitInfo:
    
    def test_git_info_returns_dict(self):
        info = get_git_info()
        
        assert isinstance(info, dict)
        assert "git_sha" in info
        assert "git_branch" in info
        assert "git_dirty" in info

class TestRunMetadata:
    
    def test_run_metadata_has_provenance_fields(self):
        metadata = RunMetadata(
            run_id="test-123",
            trainer_type="dpo",
            model_name="test-model",
            git_sha="abc123",
            git_branch="main",
            git_dirty=False,
            config_hash="hash123",
        )
        
        data = metadata.to_dict()
        
        assert data["run_id"] == "test-123"
        assert data["trainer_type"] == "dpo"
        assert data["model_name"] == "test-model"
        assert data["git_sha"] == "abc123"
        assert data["git_branch"] == "main"
        assert data["git_dirty"] == False
        assert data["config_hash"] == "hash123"
    
    def test_run_metadata_roundtrip(self):
        original = RunMetadata(
            run_id="test-123",
            trainer_type="dpo",
            config={"lr": 1e-5},
            hardware={"gpu_count": 4},
            package_versions={"torch": "2.0.0"},
        )
        
        data = original.to_dict()
        restored = RunMetadata.from_dict(data)
        
        assert restored.run_id == original.run_id
        assert restored.trainer_type == original.trainer_type
        assert restored.config == original.config
        assert restored.hardware == original.hardware
        assert restored.package_versions == original.package_versions
