"""Comprehensive test script for agent trace functionality.

This script thoroughly tests:
1. Loading traces from various formats
2. Episode filtering and manipulation
3. Diagnostics detection (loops, errors, success rates)
4. Dataset conversion for all TRL trainers (DPO, KTO, SFT, GRPO)
5. Before/after comparison workflows
6. Edge cases and error handling

Run with:
    python demo/scripts/comprehensive_agent_test.py
"""
import json
import tempfile
from pathlib import Path
import sys

# Add project root to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from post_training_toolkit.agents import (
    AgentRunLog,
    AgentTrainingLoop,
    Episode,
    Step,
    StepType,
    analyze_runs,
    to_preference_pairs,
    to_kto_dataset,
    to_sft_dataset,
    to_grpo_dataset,
    format_episode_as_conversation,
    ComparisonResult,
)


def test_trace_loading():
    """Test loading traces from various sources."""
    print("\n" + "=" * 60)
    print("TEST: Trace Loading")
    print("=" * 60)
    
    # Test 1: Load from demo JSONL
    traces_path = Path(__file__).parent.parent / "logs" / "agent_traces.jsonl"
    if traces_path.exists():
        runs = AgentRunLog.from_jsonl(traces_path)
        print(f"✓ Loaded {len(runs)} episodes from JSONL file")
        assert len(runs) > 0, "Should load at least one episode"
    
    # Test 2: Create from Episode objects
    episodes = [
        Episode(
            episode_id="test_001",
            steps=[
                Step("test_001", 0, StepType.USER_MESSAGE, content="Hello"),
                Step("test_001", 1, StepType.ASSISTANT_MESSAGE, content="Hi!"),
            ],
            success=True,
        ),
        Episode(
            episode_id="test_002",
            steps=[
                Step("test_002", 0, StepType.USER_MESSAGE, content="Fail task"),
                Step("test_002", 1, StepType.ASSISTANT_MESSAGE, content="Error"),
            ],
            success=False,
        ),
    ]
    runs_from_eps = AgentRunLog.from_episodes(episodes)
    print(f"✓ Created AgentRunLog from {len(runs_from_eps)} Episode objects")
    assert len(runs_from_eps) == 2
    
    # Test 3: Create from dict records
    records = [
        {"episode_id": "dict_001", "step": 0, "type": "user_message", "content": "Test"},
        {"episode_id": "dict_001", "step": 1, "type": "assistant_message", "content": "Response"},
        {"episode_id": "dict_001", "step": 2, "type": "episode_end", "success": True},
    ]
    runs_from_dicts = AgentRunLog.from_dicts(records)
    print(f"✓ Created AgentRunLog from {len(records)} dict records")
    assert len(runs_from_dicts) == 1
    
    # Test 4: Write and reload JSONL
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = Path(f.name)
    runs_from_eps.to_jsonl(temp_path)
    reloaded = AgentRunLog.from_jsonl(temp_path)
    print(f"✓ Round-trip to/from JSONL successful ({len(reloaded)} episodes)")
    assert len(reloaded) == 2
    temp_path.unlink()
    
    print("✓ All trace loading tests passed!")


def test_episode_properties():
    """Test Episode properties and methods."""
    print("\n" + "=" * 60)
    print("TEST: Episode Properties")
    print("=" * 60)
    
    # Create a complex episode
    episode = Episode(
        episode_id="complex_001",
        steps=[
            Step("complex_001", 0, StepType.USER_MESSAGE, content="Search and book"),
            Step("complex_001", 1, StepType.ASSISTANT_MESSAGE, content="I'll help"),
            Step("complex_001", 2, StepType.TOOL_CALL, tool="search", args={"q": "test"}),
            Step("complex_001", 3, StepType.TOOL_RESULT, tool="search", result="Found"),
            Step("complex_001", 4, StepType.TOOL_CALL, tool="search", args={"q": "test2"}),
            Step("complex_001", 5, StepType.TOOL_RESULT, tool="search", error="API error"),
            Step("complex_001", 6, StepType.TOOL_CALL, tool="book", args={"item": "x"}),
            Step("complex_001", 7, StepType.TOOL_RESULT, tool="book", result="Booked"),
            Step("complex_001", 8, StepType.ASSISTANT_MESSAGE, content="Done!"),
        ],
        success=True,
        total_tokens=500,
        total_cost=0.005,
    )
    
    # Test step accessors
    assert len(episode.user_messages) == 1, "Should have 1 user message"
    assert len(episode.assistant_messages) == 2, "Should have 2 assistant messages"
    assert len(episode.tool_calls) == 3, "Should have 3 tool calls"
    assert len(episode.tool_results) == 3, "Should have 3 tool results"
    assert len(episode.tool_errors) == 1, "Should have 1 tool error"
    print("✓ Step accessors work correctly")
    
    # Test error rate
    error_rate = episode.tool_error_rate
    assert abs(error_rate - 1/3) < 0.01, f"Tool error rate should be ~33%, got {error_rate:.1%}"
    print(f"✓ Tool error rate: {error_rate:.1%}")
    
    # Test prompt/response
    assert episode.initial_prompt == "Search and book"
    assert episode.final_response == "Done!"
    print("✓ Initial prompt and final response extraction works")
    
    # Test tool sequence
    seq = episode.get_tool_call_sequence()
    assert seq == ["search", "search", "book"], f"Got {seq}"
    print(f"✓ Tool sequence: {seq}")
    
    print("✓ All episode property tests passed!")


def test_loop_detection():
    """Test detection of repeated tool patterns."""
    print("\n" + "=" * 60)
    print("TEST: Loop Detection")
    print("=" * 60)
    
    # Episode with a clear loop (same tool 3x)
    looping_steps = []
    for i in range(3):
        looping_steps.extend([
            Step("loop_001", i*2, StepType.TOOL_CALL, tool="retry_action"),
            Step("loop_001", i*2+1, StepType.TOOL_RESULT, tool="retry_action", error="fail"),
        ])
    looping_episode = Episode(
        episode_id="loop_001",
        steps=looping_steps,
        success=False,
    )
    
    # Episode with varied tools (no loop)
    varied_episode = Episode(
        episode_id="varied_001",
        steps=[
            Step("varied_001", 0, StepType.TOOL_CALL, tool="search"),
            Step("varied_001", 1, StepType.TOOL_RESULT, tool="search", result="ok"),
            Step("varied_001", 2, StepType.TOOL_CALL, tool="filter"),
            Step("varied_001", 3, StepType.TOOL_RESULT, tool="filter", result="ok"),
            Step("varied_001", 4, StepType.TOOL_CALL, tool="book"),
            Step("varied_001", 5, StepType.TOOL_RESULT, tool="book", result="ok"),
        ],
        success=True,
    )
    
    assert looping_episode.has_repeated_tool_pattern(min_repeats=3) is True
    print("✓ Looping episode correctly detected")
    
    assert varied_episode.has_repeated_tool_pattern(min_repeats=3) is False
    print("✓ Varied episode correctly not flagged")
    
    # Test pattern loop [A, B, A, B, A, B]
    pattern_steps = []
    for i in range(3):
        pattern_steps.extend([
            Step("pattern_001", i*2, StepType.TOOL_CALL, tool="check"),
            Step("pattern_001", i*2+1, StepType.TOOL_CALL, tool="retry"),
        ])
    pattern_episode = Episode(
        episode_id="pattern_001",
        steps=pattern_steps,
        success=False,
    )
    
    assert pattern_episode.has_repeated_tool_pattern(min_repeats=3) is True
    print("✓ Pattern loop [A,B,A,B,A,B] detected")
    
    print("✓ All loop detection tests passed!")


def test_diagnostics():
    """Test diagnostics and heuristic detection."""
    print("\n" + "=" * 60)
    print("TEST: Diagnostics")
    print("=" * 60)
    
    # Create runs with various issues
    episodes = []
    
    # Successful episodes
    for i in range(5):
        episodes.append(Episode(
            episode_id=f"success_{i}",
            steps=[
                Step(f"success_{i}", 0, StepType.USER_MESSAGE, content="Task"),
                Step(f"success_{i}", 1, StepType.ASSISTANT_MESSAGE, content="Done"),
            ],
            success=True,
            total_tokens=100,
        ))
    
    # Failed episodes
    for i in range(5):
        episodes.append(Episode(
            episode_id=f"fail_{i}",
            steps=[
                Step(f"fail_{i}", 0, StepType.USER_MESSAGE, content="Task"),
                Step(f"fail_{i}", 1, StepType.TOOL_CALL, tool="action"),
                Step(f"fail_{i}", 2, StepType.TOOL_RESULT, tool="action", error="Error!"),
            ],
            success=False,
            total_tokens=200,
        ))
    
    runs = AgentRunLog.from_episodes(episodes)
    
    # Test basic stats
    assert runs.success_rate == 0.5, f"Expected 50% success rate, got {runs.success_rate:.1%}"
    print(f"✓ Success rate: {runs.success_rate:.1%}")
    
    # Run diagnostics
    report = analyze_runs(runs)
    print(f"✓ Diagnostics report generated")
    print(f"  - Total episodes: {report.total_episodes}")
    print(f"  - Success rate: {report.success_rate:.1%}")
    print(f"  - Episodes with tool errors: {report.episodes_with_tool_errors}")
    print(f"  - Insights detected: {len(report.insights)}")
    
    # Check that tool error insight was raised
    insight_types = {i.type for i in report.insights}
    assert "tool_error_spike" in insight_types, "Should detect tool error spike"
    print("✓ Tool error spike correctly detected")
    
    # Test report string formatting
    report_str = str(report)
    assert "AGENT DIAGNOSTICS REPORT" in report_str
    assert "Tool error rate" in report_str
    print("✓ Report string formatting correct")
    
    print("✓ All diagnostics tests passed!")


def test_dataset_conversion():
    """Test conversion to various TRL dataset formats."""
    print("\n" + "=" * 60)
    print("TEST: Dataset Conversion")
    print("=" * 60)
    
    # Create test episodes
    episodes = [
        Episode(
            episode_id="pos_001",
            steps=[
                Step("pos_001", 0, StepType.USER_MESSAGE, content="Task A"),
                Step("pos_001", 1, StepType.ASSISTANT_MESSAGE, content="Good response"),
            ],
            success=True,
            reward=1.0,
        ),
        Episode(
            episode_id="pos_002",
            steps=[
                Step("pos_002", 0, StepType.USER_MESSAGE, content="Task B"),
                Step("pos_002", 1, StepType.ASSISTANT_MESSAGE, content="Another good response"),
            ],
            success=True,
            reward=0.8,
        ),
        Episode(
            episode_id="neg_001",
            steps=[
                Step("neg_001", 0, StepType.USER_MESSAGE, content="Task C"),
                Step("neg_001", 1, StepType.ASSISTANT_MESSAGE, content="Bad response"),
            ],
            success=False,
            reward=0.0,
        ),
    ]
    runs = AgentRunLog.from_episodes(episodes)
    
    # Test DPO preference pairs
    print("\nTesting DPO dataset conversion...")
    dpo_dataset = to_preference_pairs(
        runs,
        positive=lambda e: e.success,
        negative=lambda e: not e.success,
    )
    assert len(dpo_dataset) == 2, f"Expected 2 pairs (2 pos × 1 neg), got {len(dpo_dataset)}"
    assert "prompt" in dpo_dataset.column_names
    assert "chosen" in dpo_dataset.column_names
    assert "rejected" in dpo_dataset.column_names
    print(f"✓ DPO dataset: {len(dpo_dataset)} preference pairs")
    
    # Test KTO dataset
    print("\nTesting KTO dataset conversion...")
    kto_dataset = to_kto_dataset(
        runs,
        desirable=lambda e: e.success,
    )
    assert len(kto_dataset) == 3
    assert "label" in kto_dataset.column_names
    labels = kto_dataset["label"]
    assert sum(labels) == 2, "Should have 2 desirable examples"
    print(f"✓ KTO dataset: {len(kto_dataset)} examples ({sum(labels)} desirable)")
    
    # Test SFT dataset
    print("\nTesting SFT dataset conversion...")
    sft_dataset = to_sft_dataset(
        runs,
        include=lambda e: e.success,
    )
    assert len(sft_dataset) == 2, "Should only include 2 successful episodes"
    assert "text" in sft_dataset.column_names
    print(f"✓ SFT dataset: {len(sft_dataset)} examples")
    
    # Test GRPO dataset
    print("\nTesting GRPO dataset conversion...")
    grpo_dataset = to_grpo_dataset(runs)
    assert len(grpo_dataset) == 3
    assert "reward" in grpo_dataset.column_names
    rewards = grpo_dataset["reward"]
    assert rewards[0] == 1.0 and rewards[2] == 0.0
    print(f"✓ GRPO dataset: {len(grpo_dataset)} examples with rewards")
    
    # Test GRPO with custom reward function
    grpo_custom = to_grpo_dataset(
        runs,
        reward_fn=lambda e: 2.0 if e.success else -1.0,
    )
    custom_rewards = grpo_custom["reward"]
    assert custom_rewards[0] == 2.0
    assert custom_rewards[2] == -1.0
    print("✓ GRPO with custom reward function works")
    
    print("\n✓ All dataset conversion tests passed!")


def test_training_loop():
    """Test the high-level AgentTrainingLoop API."""
    print("\n" + "=" * 60)
    print("TEST: AgentTrainingLoop API")
    print("=" * 60)
    
    # Create test loop
    episodes = []
    for i in range(10):
        success = i < 7  # 70% success rate
        steps_count = 5 if success else 15
        episodes.append(Episode(
            episode_id=f"ep_{i:03d}",
            steps=[
                Step(f"ep_{i:03d}", j, StepType.ASSISTANT_MESSAGE, content=f"Step {j}")
                for j in range(steps_count)
            ],
            success=success,
            total_tokens=steps_count * 50,
        ))
    
    runs = AgentRunLog.from_episodes(episodes)
    loop = AgentTrainingLoop.from_runs(runs)
    
    # Test basic properties
    assert len(loop) == 10
    assert loop.success_rate == 0.7
    print(f"✓ Loop created with {len(loop)} episodes, {loop.success_rate:.1%} success rate")
    
    # Test filtering
    successful = loop.successful()
    assert len(successful) == 7
    assert successful.success_rate == 1.0
    print(f"✓ Filter successful: {len(successful)} episodes")
    
    failed = loop.failed()
    assert len(failed) == 3
    print(f"✓ Filter failed: {len(failed)} episodes")
    
    # Test chained filtering
    short_success = loop.successful().filter(lambda e: e.total_steps < 10)
    assert len(short_success) == 7  # All successful have 5 steps
    print(f"✓ Chained filter: {len(short_success)} short successful episodes")
    
    # Test diagnostics
    report = loop.diagnose()
    assert report is not None
    assert loop.report is report  # Cached
    print("✓ Diagnostics work through loop")
    
    # Test summary
    summary = loop.summary()
    assert "episodes" in summary
    assert "success rate" in summary
    print(f"✓ Summary: {summary}")
    
    print("\n✓ All AgentTrainingLoop tests passed!")


def test_before_after_comparison():
    """Test comparing before/after training runs."""
    print("\n" + "=" * 60)
    print("TEST: Before/After Comparison")
    print("=" * 60)
    
    # "Before" training: 30% success, high step count
    before_episodes = [
        Episode(episode_id=f"before_{i}", 
                steps=[Step(f"before_{i}", j, StepType.ASSISTANT_MESSAGE, content=f"s{j}") 
                       for j in range(20)],
                success=(i < 3), total_tokens=1000)
        for i in range(10)
    ]
    before_loop = AgentTrainingLoop.from_episodes(before_episodes)
    
    # "After" training: 80% success, low step count
    after_episodes = [
        Episode(episode_id=f"after_{i}",
                steps=[Step(f"after_{i}", j, StepType.ASSISTANT_MESSAGE, content=f"s{j}")
                       for j in range(5)],
                success=(i < 8), total_tokens=250)
        for i in range(10)
    ]
    after_loop = AgentTrainingLoop.from_episodes(after_episodes)
    
    # Compare
    comparison = before_loop.compare(after_loop)
    
    # Check metrics
    assert comparison.before_success_rate == 0.3
    assert comparison.after_success_rate == 0.8
    assert comparison.success_rate_delta == 0.5
    print(f"✓ Success rate: {comparison.before_success_rate:.1%} → {comparison.after_success_rate:.1%}")
    
    assert comparison.before_avg_steps == 20
    assert comparison.after_avg_steps == 5
    assert comparison.avg_steps_delta == -15
    print(f"✓ Avg steps: {comparison.before_avg_steps:.1f} → {comparison.after_avg_steps:.1f}")
    
    # Check improved flag
    assert comparison.improved is True
    print("✓ Improvement correctly detected")
    
    # Check string output
    output = str(comparison)
    assert "BEFORE/AFTER" in output
    assert "Improvements" in output
    print("✓ Comparison string formatting correct")
    
    # Test regression case
    regression = after_loop.compare(before_loop)
    assert regression.improved is False
    assert "Regressions" in str(regression)
    print("✓ Regression case correctly identified")
    
    print("\n✓ All comparison tests passed!")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)
    
    # Empty run log
    empty_runs = AgentRunLog.from_episodes([])
    assert len(empty_runs) == 0
    assert empty_runs.success_rate == 0.0
    assert empty_runs.avg_steps == 0.0
    print("✓ Empty run log handled correctly")
    
    # Episode with no success status
    no_status_ep = Episode(
        episode_id="no_status",
        steps=[Step("no_status", 0, StepType.USER_MESSAGE, content="Test")],
        success=None,
    )
    runs_no_status = AgentRunLog.from_episodes([no_status_ep])
    assert runs_no_status.success_rate == 0.0  # No episodes with status
    print("✓ Episodes without success status handled")
    
    # Episode with no tools
    no_tools_ep = Episode(
        episode_id="no_tools",
        steps=[
            Step("no_tools", 0, StepType.USER_MESSAGE, content="Test"),
            Step("no_tools", 1, StepType.ASSISTANT_MESSAGE, content="Response"),
        ],
        success=True,
    )
    assert no_tools_ep.tool_error_rate == 0.0
    assert no_tools_ep.has_repeated_tool_pattern() is False
    print("✓ Episodes without tools handled")
    
    # Test error on no matching predicates
    runs = AgentRunLog.from_episodes([
        Episode(episode_id="ep1", steps=[], success=True),
    ])
    try:
        to_preference_pairs(
            runs,
            positive=lambda e: e.success,
            negative=lambda e: not e.success,  # No failed episodes
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No episodes matched" in str(e)
        print("✓ Correct error on no negative examples")
    
    print("\n✓ All edge case tests passed!")


def test_format_functions():
    """Test conversation formatting functions."""
    print("\n" + "=" * 60)
    print("TEST: Formatting Functions")
    print("=" * 60)
    
    episode = Episode(
        episode_id="fmt_001",
        steps=[
            Step("fmt_001", 0, StepType.USER_MESSAGE, content="Find flights"),
            Step("fmt_001", 1, StepType.ASSISTANT_MESSAGE, content="Searching..."),
            Step("fmt_001", 2, StepType.TOOL_CALL, tool="search", args={"dest": "Paris"}),
            Step("fmt_001", 3, StepType.TOOL_RESULT, tool="search", result="Found 3 flights"),
            Step("fmt_001", 4, StepType.ASSISTANT_MESSAGE, content="Here are your options"),
        ],
        success=True,
    )
    
    # Default formatting
    conv = format_episode_as_conversation(episode)
    assert "User: Find flights" in conv
    assert "Assistant:" in conv
    assert "[Tool Call: search" in conv
    assert "[Tool Result:" in conv
    print("✓ Default formatting includes all step types")
    
    # Without tool calls
    conv_no_tools = format_episode_as_conversation(
        episode, include_tool_calls=False, include_tool_results=False
    )
    assert "[Tool Call" not in conv_no_tools
    assert "[Tool Result" not in conv_no_tools
    assert "User: Find flights" in conv_no_tools
    print("✓ Formatting without tools works")
    
    # With max steps limit
    conv_limited = format_episode_as_conversation(episode, max_steps=2)
    lines = conv_limited.strip().split('\n')
    assert len(lines) == 2
    print("✓ max_steps limiting works")
    
    print("\n✓ All formatting tests passed!")


def main():
    """Run all comprehensive tests."""
    print("=" * 60)
    print("COMPREHENSIVE AGENT FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        test_trace_loading,
        test_episode_properties,
        test_loop_detection,
        test_diagnostics,
        test_dataset_conversion,
        test_training_loop,
        test_before_after_comparison,
        test_edge_cases,
        test_format_functions,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All comprehensive tests passed!")
        print("The agent functionality is working correctly.")


if __name__ == "__main__":
    main()
