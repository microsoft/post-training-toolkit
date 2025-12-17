"""Tests for snapshot capture utilities."""

from pathlib import Path

from post_training_toolkit.models.artifacts import RunArtifactManager
from post_training_toolkit.models.snapshots import SnapshotManager


class DummyTokenizer:
    """Minimal tokenizer that tokenizes on whitespace."""

    def encode(self, text, add_special_tokens=False):
        return text.strip().split()


class DummyModel:
    """Minimal model stub with training mode tracking."""

    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode
        return self


def test_token_length_uses_tokens(tmp_path: Path):
    """Snapshot output_length should use token counts, not character counts."""
    artifact_manager = RunArtifactManager(tmp_path, is_main_process_override=True)
    artifact_manager.initialize()

    prompts = ["hello world"]
    outputs = ["one two three four"]  # 4 tokens

    manager = SnapshotManager(
        artifact_manager=artifact_manager,
        prompts=prompts,
        generate_fn=lambda m, t, p, cfg: outputs,
        snapshot_interval=1,
        compute_scores=False,
    )

    snapshot = manager.capture(step=0, model=DummyModel(), tokenizer=DummyTokenizer())
    assert snapshot is not None
    assert snapshot.entries[0].output_length == 4


def test_preserve_training_mode_context():
    """Training mode should be restored after the preservation context."""
    model = DummyModel()
    assert model.training is True

    with SnapshotManager._preserve_training_mode(model):
        assert model.training is False

    assert model.training is True
