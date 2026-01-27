import json
import tempfile
from pathlib import Path

import pytest

from post_training_toolkit.agents import (
    AgentRunLog,
    Episode,
    Step,
    StepType,
    analyze_runs,
    AgentDiagnosticsReport,
    to_preference_pairs,
    to_kto_dataset,
    format_episode_as_conversation,
)

SAMPLE_TRACES = [
    {"episode_id": "ep_001", "step": 0, "type": "user_message", "content": "Hello"},
    {"episode_id": "ep_001", "step": 1, "type": "assistant_message", "content": "Hi there!"},
    {"episode_id": "ep_001", "step": 2, "type": "episode_end", "success": True, "reward": 1.0, "total_tokens": 50},
    {"episode_id": "ep_002", "step": 0, "type": "user_message", "content": "Search for X"},
    {"episode_id": "ep_002", "step": 1, "type": "tool_call", "tool": "search", "args": {"q": "X"}},
    {"episode_id": "ep_002", "step": 2, "type": "tool_result", "tool": "search", "error": "API error"},
    {"episode_id": "ep_002", "step": 3, "type": "episode_end", "success": False, "reward": 0.0, "total_tokens": 100},
    {"episode_id": "ep_003", "step": 0, "type": "user_message", "content": "Do task"},
    {"episode_id": "ep_003", "step": 1, "type": "tool_call", "tool": "action", "args": {}},
    {"episode_id": "ep_003", "step": 2, "type": "tool_result", "tool": "action", "result": "fail"},
    {"episode_id": "ep_003", "step": 3, "type": "tool_call", "tool": "action", "args": {}},
    {"episode_id": "ep_003", "step": 4, "type": "tool_result", "tool": "action", "result": "fail"},
    {"episode_id": "ep_003", "step": 5, "type": "tool_call", "tool": "action", "args": {}},
    {"episode_id": "ep_003", "step": 6, "type": "tool_result", "tool": "action", "result": "fail"},
    {"episode_id": "ep_003", "step": 7, "type": "episode_end", "success": False, "total_tokens": 200},
]

@pytest.fixture
def sample_traces_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in SAMPLE_TRACES:
            f.write(json.dumps(record) + '\n')
        return Path(f.name)

@pytest.fixture
def sample_runs(sample_traces_file):
    return AgentRunLog.from_jsonl(sample_traces_file)

class TestAgentRunLog:
    
    def test_load_from_jsonl(self, sample_runs):
        assert len(sample_runs) == 3
        assert sample_runs.success_rate == pytest.approx(1/3)
    
    def test_episode_access(self, sample_runs):
        ep = sample_runs["ep_001"]
        assert ep.episode_id == "ep_001"
        assert ep.success is True
        
        ep0 = sample_runs[0]
        assert ep0.episode_id == "ep_001"
    
    def test_filter(self, sample_runs):
        successful = sample_runs.filter(lambda e: e.success is True)
        assert len(successful) == 1
        assert successful[0].episode_id == "ep_001"
    
    def test_split(self, sample_runs):
        good, bad = sample_runs.split(lambda e: e.success is True)
        assert len(good) == 1
        assert len(bad) == 2

class TestEpisode:
    
    def test_step_accessors(self, sample_runs):
        ep = sample_runs["ep_002"]
        assert len(ep.tool_calls) == 1
        assert len(ep.tool_results) == 1
        assert len(ep.tool_errors) == 1
        assert ep.tool_error_rate == 1.0
    
    def test_initial_prompt(self, sample_runs):
        ep = sample_runs["ep_001"]
        assert ep.initial_prompt == "Hello"
    
    def test_loop_detection(self, sample_runs):
        ep_normal = sample_runs["ep_001"]
        assert ep_normal.has_repeated_tool_pattern() is False
        
        ep_loop = sample_runs["ep_003"]
        assert ep_loop.has_repeated_tool_pattern(min_repeats=3) is True
    
    def test_tool_sequence(self, sample_runs):
        ep = sample_runs["ep_003"]
        seq = ep.get_tool_call_sequence()
        assert seq == ["action", "action", "action"]

class TestAnalyzeRuns:
    
    def test_basic_report(self, sample_runs):
        report = analyze_runs(sample_runs)
        
        assert report.total_episodes == 3
        assert report.success_rate == pytest.approx(1/3)
        assert report.episodes_with_loops == 1
        assert report.episodes_with_tool_errors == 1
    
    def test_insights_detected(self, sample_runs):
        report = analyze_runs(sample_runs)
        
        insight_types = {i.type for i in report.insights}
        assert "loop_detected" in insight_types or "tool_error_spike" in insight_types
    
    def test_report_string(self, sample_runs):
        report = analyze_runs(sample_runs)
        report_str = str(report)
        
        assert "AGENT DIAGNOSTICS REPORT" in report_str
        assert "Episodes analyzed:" in report_str

class TestConverters:
    
    def test_format_episode_as_conversation(self, sample_runs):
        ep = sample_runs["ep_001"]
        conv = format_episode_as_conversation(ep)
        
        assert "User: Hello" in conv
        assert "Assistant: Hi there!" in conv
    
    def test_to_preference_pairs(self, sample_runs):
        dataset = to_preference_pairs(
            sample_runs,
            positive=lambda e: e.success is True,
            negative=lambda e: e.success is False,
        )
        
        assert len(dataset) == 2
        assert "prompt" in dataset.column_names
        assert "chosen" in dataset.column_names
        assert "rejected" in dataset.column_names
    
    def test_to_preference_pairs_no_match(self, sample_runs):
        with pytest.raises(ValueError, match="No episodes matched"):
            to_preference_pairs(
                sample_runs,
                positive=lambda e: False,
                negative=lambda e: e.success is False,
            )
    
    def test_to_kto_dataset(self, sample_runs):
        dataset = to_kto_dataset(
            sample_runs,
            desirable=lambda e: e.success is True,
        )
        
        assert len(dataset) == 3
        assert "prompt" in dataset.column_names
        assert "completion" in dataset.column_names
        assert "label" in dataset.column_names
        
        labels = dataset["label"]
        assert sum(labels) == 1

class TestStepType:
    
    def test_from_str_known(self):
        assert StepType.from_str("user_message") == StepType.USER_MESSAGE
        assert StepType.from_str("tool_call") == StepType.TOOL_CALL
    
    def test_from_str_unknown(self):
        assert StepType.from_str("custom_type") == StepType.OTHER
