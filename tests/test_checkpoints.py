"""Smoke tests for checkpoint comparison and resume validation."""
import pytest
import tempfile
import json
from pathlib import Path

from post_training_toolkit.models.checkpoints import (
    CheckpointComparator,
    CheckpointScore,
    CheckpointRecommendation,
    ResumeValidator,
    ResumeValidationResult,
    recommend_checkpoint,
    validate_resume,
    compute_metric_stability,
    compute_snapshot_consistency,
)
from post_training_toolkit.models.artifacts import (
    RunArtifactManager,
    Snapshot,
    SnapshotEntry,
    SnapshotMetadata,
)


class TestCheckpointScore:
    """Tests for CheckpointScore dataclass."""
    
    def test_score_creation(self):
        score = CheckpointScore(
            step=100,
            stability_score=0.8,
            drift_score=0.2,
            refusal_rate=0.05,
            length_consistency=0.9,
            entropy_consistency=0.85,
            overall_score=0.75,
        )
        assert score.step == 100
        assert score.overall_score == 0.75
    
    def test_score_to_dict(self):
        score = CheckpointScore(
            step=100,
            stability_score=0.8,
            drift_score=0.2,
            refusal_rate=0.05,
            length_consistency=0.9,
            entropy_consistency=0.85,
            overall_score=0.75,
        )
        d = score.to_dict()
        assert d["step"] == 100
        assert d["overall"] == 0.75
        assert "stability" in d
        assert "drift" in d


class TestMetricStability:
    """Tests for metric stability computation."""
    
    def test_stable_metrics(self):
        """Stable metrics should have high stability score."""
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        df = pd.DataFrame({
            "step": range(100),
            "loss": np.random.normal(0.5, 0.01, 100),  # Low variance
        })
        
        stability = compute_metric_stability(df, step=50, loss_key="loss")
        assert stability > 0.5  # Should be relatively stable
    
    def test_unstable_metrics(self):
        """Unstable metrics should have lower stability score."""
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({
            "step": range(100),
            "loss": [0.1 if i % 2 == 0 else 2.0 for i in range(100)],  # High variance
        })
        
        stability = compute_metric_stability(df, step=50, loss_key="loss")
        assert stability < 0.8  # Should be less stable


class TestSnapshotConsistency:
    """Tests for snapshot consistency computation."""
    
    def test_consistent_snapshot(self):
        """Consistent outputs should have high consistency scores."""
        entries = [
            SnapshotEntry(
                prompt_id=f"prompt_{i}",
                prompt="Test prompt",
                output="Test output " * 10,
                output_length=100,
                is_refusal=False,
                entropy_mean=2.5,
            )
            for i in range(10)
        ]
        metadata = SnapshotMetadata(
            step=100,
            timestamp="2025-01-01T00:00:00Z",
            num_prompts=10,
        )
        snapshot = Snapshot(metadata=metadata, entries=entries)
        
        length_cons, entropy_cons = compute_snapshot_consistency(snapshot)
        assert length_cons > 0.9  # All same length
        assert entropy_cons > 0.9  # All same entropy
    
    def test_inconsistent_snapshot(self):
        """Inconsistent outputs should have lower consistency scores."""
        entries = [
            SnapshotEntry(
                prompt_id=f"prompt_{i}",
                prompt="Test prompt",
                output="x" * (i * 50),  # Varying lengths
                output_length=i * 50,
                is_refusal=False,
                entropy_mean=i * 0.5,  # Varying entropy
            )
            for i in range(1, 11)
        ]
        metadata = SnapshotMetadata(
            step=100,
            timestamp="2025-01-01T00:00:00Z",
            num_prompts=10,
        )
        snapshot = Snapshot(metadata=metadata, entries=entries)
        
        length_cons, entropy_cons = compute_snapshot_consistency(snapshot)
        assert length_cons < 0.9
        assert entropy_cons < 0.9


class TestResumeValidation:
    """Tests for resume validation."""
    
    @pytest.fixture
    def temp_run_dir(self):
        """Create a temporary run directory with metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # Create metrics file
            metrics_path = run_dir / "metrics.jsonl"
            with open(metrics_path, "w") as f:
                f.write(json.dumps({"type": "header", "trainer_type": "dpo"}) + "\n")
                for step in range(100):
                    f.write(json.dumps({"step": step, "metrics": {"loss": 0.5}}) + "\n")
            
            # Create metadata
            metadata_path = run_dir / "run_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump({
                    "trainer_type": "dpo",
                    "config": {
                        "learning_rate": 1e-5,
                        "beta": 0.1,
                    }
                }, f)
            
            yield run_dir
    
    def test_valid_resume(self, temp_run_dir):
        """Valid resume should pass validation."""
        result = validate_resume(
            run_dir=temp_run_dir,
            checkpoint_step=99,
        )
        
        assert result.is_valid
        assert result.resumed_from_step == 99
        assert result.expected_next_step == 100
        assert len(result.errors) == 0
    
    def test_resume_from_future_step(self, temp_run_dir):
        """Resume from step ahead of logs should fail."""
        result = validate_resume(
            run_dir=temp_run_dir,
            checkpoint_step=150,  # Ahead of logged steps
        )
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("ahead" in e.lower() for e in result.errors)
    
    def test_resume_from_past_step(self, temp_run_dir):
        """Resume from earlier step should warn about existing metrics."""
        result = validate_resume(
            run_dir=temp_run_dir,
            checkpoint_step=50,  # Behind logged steps
        )
        
        # Should be valid but with warnings
        assert len(result.warnings) > 0
    
    def test_resume_validation_result_to_dict(self):
        """ResumeValidationResult should serialize to dict."""
        result = ResumeValidationResult(
            is_valid=True,
            resumed_from_step=100,
            expected_next_step=101,
            actual_next_step=None,
            warnings=["Test warning"],
            errors=[],
            checkpoint_path="/path/to/checkpoint",
            state_hash="abc123",
        )
        
        d = result.to_dict()
        assert d["is_valid"] == True
        assert d["resumed_from_step"] == 100
        assert d["warnings"] == ["Test warning"]


class TestResumeValidator:
    """Tests for ResumeValidator class."""
    
    @pytest.fixture
    def temp_run_dir(self):
        """Create a temporary run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # Create metrics file
            metrics_path = run_dir / "metrics.jsonl"
            with open(metrics_path, "w") as f:
                for step in range(50):
                    f.write(json.dumps({"step": step, "metrics": {"loss": 0.5}}) + "\n")
            
            yield run_dir
    
    def test_verify_first_step_correct(self, temp_run_dir):
        """Correct first step should pass verification."""
        artifact_manager = RunArtifactManager(temp_run_dir)
        validator = ResumeValidator(artifact_manager)
        
        # Validate resume from step 49
        validator.validate_resume(checkpoint_step=49)
        
        # Verify first step is 50 (correct)
        is_valid, errors = validator.verify_first_step(step=50)
        assert is_valid
        assert len(errors) == 0
    
    def test_verify_first_step_incorrect(self, temp_run_dir):
        """Incorrect first step should fail verification."""
        artifact_manager = RunArtifactManager(temp_run_dir)
        validator = ResumeValidator(artifact_manager)
        
        # Validate resume from step 49
        validator.validate_resume(checkpoint_step=49)
        
        # Verify first step is 60 (incorrect, should be 50)
        is_valid, errors = validator.verify_first_step(step=60)
        assert not is_valid
        assert len(errors) > 0
        assert any("mismatch" in e.lower() for e in errors)
    
    def test_config_validation(self, temp_run_dir):
        """Config changes should generate warnings."""
        # Create metadata with original config
        metadata_path = temp_run_dir / "run_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "config": {
                    "learning_rate": 1e-5,
                    "beta": 0.1,
                }
            }, f)
        
        artifact_manager = RunArtifactManager(temp_run_dir)
        validator = ResumeValidator(artifact_manager)
        
        # Resume with different config
        result = validator.validate_resume(
            checkpoint_step=49,
            config={
                "learning_rate": 5e-5,  # Changed!
                "beta": 0.1,
            }
        )
        
        assert len(result.warnings) > 0
        assert any("learning_rate" in w for w in result.warnings)
