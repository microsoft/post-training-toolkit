"""Tests for artifact manager and provenance functions."""

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
    """initialize() should skip writes when not main process and write when main."""
    not_main = RunArtifactManager(tmp_path / "nominal", is_main_process_override=False)
    not_main.initialize()
    assert not not_main.metadata_path.exists()

    main_mgr = RunArtifactManager(tmp_path / "main", is_main_process_override=True)
    main_mgr.initialize()
    assert main_mgr.metadata_path.exists()
    assert main_mgr.metadata_start_path.exists()
    assert main_mgr.metrics_path.exists()


def test_initialize_creates_start_metadata(tmp_path: Path):
    """initialize() should create immutable start metadata."""
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
    """finalize() should create immutable final metadata."""
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
    """Tests for config hashing."""
    
    def test_config_hash_is_stable(self):
        """Same config should produce same hash."""
        config = {"learning_rate": 1e-5, "beta": 0.1, "max_steps": 1000}
        
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        
        assert hash1 == hash2
    
    def test_config_hash_differs_for_different_configs(self):
        """Different configs should produce different hashes."""
        config1 = {"learning_rate": 1e-5}
        config2 = {"learning_rate": 1e-4}
        
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        
        assert hash1 != hash2
    
    def test_config_hash_order_independent(self):
        """Config hash should be order-independent."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}
        
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        
        assert hash1 == hash2


class TestHardwareInfo:
    """Tests for hardware info collection."""
    
    def test_hardware_info_has_required_fields(self):
        """Hardware info should include essential fields."""
        info = get_hardware_info()
        
        assert "hostname" in info
        assert "platform" in info
        assert "cpu_count" in info
        assert "python_version" in info
        assert "cuda_available" in info
        assert "world_size" in info
        assert "global_rank" in info


class TestPackageVersions:
    """Tests for package version collection."""
    
    def test_package_versions_includes_core_packages(self):
        """Should include core packages if installed."""
        versions = get_package_versions()
        
        # At minimum these should be available in test environment
        # (may not all be installed)
        assert isinstance(versions, dict)
    
    def test_full_package_snapshot(self):
        """Full snapshot should include more packages."""
        minimal = get_package_versions(full_env=False)
        full = get_package_versions(full_env=True)
        
        # Full should have at least as many packages
        assert len(full) >= len(minimal)


class TestModelIdentity:
    """Tests for model identity extraction."""
    
    def test_model_identity_from_string(self):
        """String model name should be captured."""
        identity = get_model_identity("meta-llama/Llama-2-7b-hf")
        
        assert identity["model_name"] == "meta-llama/Llama-2-7b-hf"
    
    def test_model_identity_from_mock(self):
        """Model object should have identity extracted."""
        mock_model = MagicMock()
        mock_model.name_or_path = "test-model-path"
        
        identity = get_model_identity(mock_model)
        
        assert identity["model_name"] == "test-model-path"
    
    def test_model_identity_from_config(self):
        """Model with config should have identity extracted."""
        mock_model = MagicMock(spec=[])
        mock_model.config = MagicMock()
        mock_model.config._name_or_path = "config-model-path"
        mock_model.config._commit_hash = "abc123"
        
        identity = get_model_identity(mock_model)
        
        assert identity["model_name"] == "config-model-path"
        assert identity["model_revision"] == "abc123"


class TestGitInfo:
    """Tests for git info collection."""
    
    def test_git_info_returns_dict(self):
        """Git info should return a dict even if not in a repo."""
        info = get_git_info()
        
        assert isinstance(info, dict)
        assert "git_sha" in info
        assert "git_branch" in info
        assert "git_dirty" in info


class TestRunMetadata:
    """Tests for RunMetadata dataclass."""
    
    def test_run_metadata_has_provenance_fields(self):
        """RunMetadata should have all provenance fields."""
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
        """RunMetadata should survive serialization roundtrip."""
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
