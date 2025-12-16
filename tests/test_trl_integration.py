#!/usr/bin/env python3
"""
Real TRL Integration Tests - Actually runs trainers on CPU.

These tests verify the diagnostics callback works with real TRL trainers,
not mocks. They run on CPU with tiny models so they complete in seconds.

Usage:
    pytest tests/test_trl_integration.py -v
    # or directly:
    python tests/test_trl_integration.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# Check if TRL dependencies are available
try:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    HAS_TRL = True
except ImportError:
    HAS_TRL = False

# Check for SFT
try:
    from trl import SFTConfig, SFTTrainer
    HAS_SFT = True
except ImportError:
    HAS_SFT = False


# Tiny model that loads fast
TINY_MODEL = "hf-internal-testing/tiny-random-GPT2LMHeadModel"


def create_preference_dataset(n_samples: int = 10) -> Dataset:
    """Create minimal preference dataset for DPO."""
    return Dataset.from_dict({
        "prompt": [f"Question {i}:" for i in range(n_samples)],
        "chosen": [f" Good answer {i}." for i in range(n_samples)],
        "rejected": [f" Bad {i}." for i in range(n_samples)],
    })


def create_sft_dataset(n_samples: int = 10) -> Dataset:
    """Create minimal dataset for SFT."""
    return Dataset.from_dict({
        "text": [f"Example text number {i}. This is a sample." for i in range(n_samples)],
    })


@pytest.mark.skipif(not HAS_TRL, reason="TRL not installed")
class TestDPOIntegration:
    """Real DPO trainer integration tests."""
    
    def test_dpo_trainer_metrics_captured(self):
        """Verify DiagnosticsCallback captures metrics from real DPOTrainer."""
        from post_training_toolkit import DiagnosticsCallback
        
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            metrics_path = run_dir / "metrics.jsonl"
            
            # Load tiny model
            tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
            
            # Create callback
            callback = DiagnosticsCallback(
                run_dir=str(run_dir),
                verbose=True,
            )
            
            # DPO config for fast CPU run
            config = DPOConfig(
                output_dir=str(run_dir / "checkpoints"),
                max_steps=3,
                per_device_train_batch_size=2,
                logging_steps=1,
                report_to="none",
                use_cpu=True,
                remove_unused_columns=False,
            )
            
            # Create and run trainer
            trainer = DPOTrainer(
                model=model,
                args=config,
                train_dataset=create_preference_dataset(),
                processing_class=tokenizer,
                callbacks=[callback],
            )
            
            trainer.train()
            
            # Assertions
            assert callback.trainer_type == "dpo", f"Expected 'dpo', got '{callback.trainer_type}'"
            assert metrics_path.exists(), "metrics.jsonl was not created"
            
            # Check metrics content
            with open(metrics_path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            
            # Should have header + at least one data line
            assert len(lines) >= 2, f"Expected at least 2 lines, got {len(lines)}"
            
            header = lines[0]
            assert header.get("type") == "header", "First line should be header"
            assert header.get("trainer_type") == "dpo", f"Header trainer_type: {header.get('trainer_type')}"
            
            # Check that some metrics were logged (data lines have "step" key, no "type")
            data_lines = [l for l in lines if "step" in l and l.get("type") != "header"]
            assert len(data_lines) >= 1, "No metrics data lines found"
            
            # Check that metrics dict is present
            assert "metrics" in data_lines[0], "Missing 'metrics' dict in data line"
            metrics_dict = data_lines[0]["metrics"]
            assert len(metrics_dict) > 0, "Metrics dict is empty"
            
            print(f"\n✓ DPO integration test passed")
            print(f"  Trainer type: {callback.trainer_type}")
            print(f"  Metrics logged: {len(data_lines)} data lines")
            print(f"  Sample metrics: {list(metrics_dict.keys())[:10]}")
    
    def test_dpo_metadata_created(self):
        """Verify run_metadata.json is created with correct content."""
        from post_training_toolkit import DiagnosticsCallback
        
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
            
            callback = DiagnosticsCallback(run_dir=str(run_dir))
            
            config = DPOConfig(
                output_dir=str(run_dir / "checkpoints"),
                max_steps=2,
                per_device_train_batch_size=2,
                logging_steps=1,
                report_to="none",
                use_cpu=True,
                remove_unused_columns=False,
            )
            
            trainer = DPOTrainer(
                model=model,
                args=config,
                train_dataset=create_preference_dataset(),
                processing_class=tokenizer,
                callbacks=[callback],
            )
            
            trainer.train()
            
            # Check metadata
            metadata_path = run_dir / "run_metadata.json"
            assert metadata_path.exists(), "run_metadata.json not created"
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert metadata.get("trainer_type") == "dpo"
            assert metadata.get("status") == "completed"
            assert "start_time" in metadata
            assert "end_time" in metadata
            
            print(f"\n✓ Metadata test passed")
            print(f"  Status: {metadata.get('status')}")


@pytest.mark.skipif(not HAS_SFT, reason="SFTTrainer not available")
class TestSFTIntegration:
    """Real SFT trainer integration tests."""
    
    def test_sft_trainer_detection(self):
        """Verify SFTTrainer is correctly detected."""
        from post_training_toolkit import DiagnosticsCallback
        
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
            
            callback = DiagnosticsCallback(run_dir=str(run_dir))
            
            config = SFTConfig(
                output_dir=str(run_dir / "checkpoints"),
                max_steps=2,
                per_device_train_batch_size=2,
                logging_steps=1,
                report_to="none",
                use_cpu=True,
            )
            
            trainer = SFTTrainer(
                model=model,
                args=config,
                train_dataset=create_sft_dataset(),
                processing_class=tokenizer,
                callbacks=[callback],
            )
            
            trainer.train()
            
            assert callback.trainer_type == "sft", f"Expected 'sft', got '{callback.trainer_type}'"
            print(f"\n✓ SFT detection test passed")


class TestDiagnosticsEngine:
    """Test the diagnostics engine on generated output."""
    
    @pytest.mark.skipif(not HAS_TRL, reason="TRL not installed")
    def test_diagnostics_on_dpo_output(self):
        """Run diagnostics engine on DPO training output."""
        from post_training_toolkit import DiagnosticsCallback, run_diagnostics
        
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # First run DPO to generate metrics
            tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
            
            callback = DiagnosticsCallback(run_dir=str(run_dir))
            
            config = DPOConfig(
                output_dir=str(run_dir / "checkpoints"),
                max_steps=5,
                per_device_train_batch_size=2,
                logging_steps=1,
                report_to="none",
                use_cpu=True,
                remove_unused_columns=False,
            )
            
            trainer = DPOTrainer(
                model=model,
                args=config,
                train_dataset=create_preference_dataset(),
                processing_class=tokenizer,
                callbacks=[callback],
            )
            
            trainer.train()
            
            # Now run diagnostics
            reports_dir = run_dir / "reports"
            report_path = run_diagnostics(
                run_dir / "metrics.jsonl",
                reports_dir,
                make_plots=False,  # Skip plots for speed
            )
            
            assert report_path.exists(), "Diagnostics report not generated"
            
            content = report_path.read_text()
            assert len(content) > 100, "Report seems too short"
            assert "dpo" in content.lower() or "DPO" in content, "Report should mention DPO"
            
            print(f"\n✓ Diagnostics engine test passed")
            print(f"  Report: {report_path}")
            print(f"  Report size: {len(content)} chars")


class TestCLI:
    """Test CLI functionality."""
    
    def test_cli_help(self):
        """Verify CLI --help works."""
        import subprocess
        result = subprocess.run(
            ["python", "-m", "post_training_toolkit.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI --help failed: {result.stderr}"
        assert "ptt-diagnose" in result.stdout or "usage" in result.stdout.lower()
        print(f"\n✓ CLI help test passed")


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("TRL Integration Tests")
    print("=" * 60)
    
    if not HAS_TRL:
        print("\n⚠ TRL not installed - skipping integration tests")
        print("  Install with: pip install trl transformers datasets torch")
        return 1
    
    import traceback
    
    tests = [
        ("DPO metrics captured", TestDPOIntegration().test_dpo_trainer_metrics_captured),
        ("DPO metadata created", TestDPOIntegration().test_dpo_metadata_created),
        ("Diagnostics engine", TestDiagnosticsEngine().test_diagnostics_on_dpo_output),
        ("CLI help", TestCLI().test_cli_help),
    ]
    
    if HAS_SFT:
        tests.append(("SFT detection", TestSFTIntegration().test_sft_trainer_detection))
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(run_all_tests())
