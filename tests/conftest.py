"""Pytest configuration and shared fixtures."""
import pytest
import tempfile
import json
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_metrics_file(temp_dir):
    """Create a sample metrics JSONL file."""
    metrics_path = temp_dir / "metrics.jsonl"
    
    with open(metrics_path, "w") as f:
        # Header
        f.write(json.dumps({
            "type": "header",
            "trainer_type": "dpo",
            "timestamp": "2025-01-01T00:00:00Z"
        }) + "\n")
        
        # Metrics
        for step in range(100):
            f.write(json.dumps({
                "step": step,
                "timestamp": f"2025-01-01T00:{step:02d}:00Z",
                "trainer_type": "dpo",
                "metrics": {
                    "dpo_loss": 0.693 - (step * 0.002),
                    "reward_mean": 0.1 + (step * 0.005),
                    "win_rate": 0.5 + (step * 0.002),
                    "kl": 0.05 + (step * 0.001),
                }
            }) + "\n")
        
        # Footer
        f.write(json.dumps({
            "type": "footer",
            "total_steps": 100,
            "timestamp": "2025-01-01T01:40:00Z"
        }) + "\n")
    
    return metrics_path


@pytest.fixture
def sample_run_dir(temp_dir, sample_metrics_file):
    """Create a complete sample run directory."""
    # Create subdirectories
    (temp_dir / "snapshots").mkdir()
    (temp_dir / "checkpoints").mkdir()
    
    # Create run metadata
    metadata_path = temp_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "trainer_type": "dpo",
            "start_time": "2025-01-01T00:00:00Z",
            "config": {
                "learning_rate": 1e-5,
                "beta": 0.1,
                "batch_size": 4,
            }
        }, f)
    
    return temp_dir
