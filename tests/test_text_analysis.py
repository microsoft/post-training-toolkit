import pytest
from pathlib import Path
from post_training_toolkit.models.artifacts import Snapshot, SnapshotMetadata, SnapshotEntry
from post_training_toolkit.models.text_analysis import (
    detect_verbosity_bias,
    detect_repetition_loops,
    detect_pattern_collapse
)

def create_snapshot(step, outputs):
    entries = [
        SnapshotEntry(
            prompt_id=f"p{i}",
            prompt=f"prompt {i}",
            output=out,
            output_length=len(out),
            is_refusal=False
        ) for i, out in enumerate(outputs)
    ]
    return Snapshot(
        metadata=SnapshotMetadata(step=step, timestamp="", num_prompts=len(outputs)),
        entries=entries
    )

def test_detect_verbosity_bias():
    # Case 1: No bias
    snapshots = [
        create_snapshot(10, ["a" * 10]),
        create_snapshot(20, ["a" * 10]),
        create_snapshot(30, ["a" * 10]),
    ]
    assert len(detect_verbosity_bias(snapshots)) == 0
    
    # Case 2: Strong bias (doubling length)
    snapshots = [
        create_snapshot(10, ["a" * 10]),
        create_snapshot(20, ["a" * 20]),
        create_snapshot(30, ["a" * 30]),
    ]
    insights = detect_verbosity_bias(snapshots)
    assert len(insights) == 1
    assert insights[0].type == "verbosity_bias"

def test_detect_repetition_loops():
    # Case 1: Normal text
    snapshots = [create_snapshot(10, ["The quick brown fox jumps over the lazy dog."])]
    assert len(detect_repetition_loops(snapshots)) == 0
    
    # Case 2: Repetitive text
    repetitive = "I apologize. " * 50
    snapshots = [create_snapshot(10, [repetitive])]
    insights = detect_repetition_loops(snapshots)
    assert len(insights) == 1
    assert insights[0].type == "repetition_collapse"

def test_detect_pattern_collapse():
    # Case 1: Diverse starts
    snapshots = [create_snapshot(10, ["Hello there", "What is this", "I think that"])]
    assert len(detect_pattern_collapse(snapshots)) == 0
    
    # Case 2: Collapsed starts
    collapsed = ["As an AI language model" + str(i) for i in range(10)]
    snapshots = [create_snapshot(10, collapsed)]
    insights = detect_pattern_collapse(snapshots)
    assert len(insights) == 1
    assert insights[0].type == "pattern_collapse"
