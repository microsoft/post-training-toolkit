import pytest
from pathlib import Path

from post_training_toolkit.agents import (
    AgentTrainingLoop,
    AgentRunLog,
    Episode,
    Step,
    StepType,
    ComparisonResult,
)

def make_episode(episode_id: str, success: bool, steps: int = 5) -> Episode:
    step_list = [
        Step(episode_id=episode_id, step=0, type=StepType.USER_MESSAGE, content="Do task"),
    ]
    for i in range(1, steps):
        step_list.append(
            Step(episode_id=episode_id, step=i, type=StepType.ASSISTANT_MESSAGE, content=f"Step {i}")
        )
    return Episode(
        episode_id=episode_id,
        steps=step_list,
        success=success,
        total_tokens=steps * 100,
    )

@pytest.fixture
def sample_runs() -> AgentRunLog:
    episodes = [
        make_episode("ep_001", success=True, steps=5),
        make_episode("ep_002", success=True, steps=8),
        make_episode("ep_003", success=False, steps=15),
        make_episode("ep_004", success=False, steps=20),
        make_episode("ep_005", success=True, steps=3),
    ]
    return AgentRunLog.from_episodes(episodes)

@pytest.fixture
def sample_jsonl(sample_runs, tmp_path) -> Path:
    path = tmp_path / "traces.jsonl"
    sample_runs.to_jsonl(path)
    return path

class TestAgentTrainingLoopCreation:
    
    def test_from_runs(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        assert len(loop) == 5
    
    def test_from_traces(self, sample_jsonl):
        loop = AgentTrainingLoop.from_traces(sample_jsonl)
        assert len(loop) == 5
    
    def test_from_episodes(self):
        episodes = [make_episode("ep_001", True), make_episode("ep_002", False)]
        loop = AgentTrainingLoop.from_episodes(episodes)
        assert len(loop) == 2
    
    def test_access_runs(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        assert loop.runs is sample_runs
        assert len(loop.episodes) == 5

class TestAgentTrainingLoopDiagnose:
    
    def test_diagnose_returns_report(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        report = loop.diagnose()
        
        assert report.total_episodes == 5
        assert report.success_rate == 0.6
        assert loop.report is report
    
    def test_diagnose_with_budget(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        report = loop.diagnose(budget_per_episode=0.01)
        assert report is not None
    
    def test_report_is_none_before_diagnose(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        assert loop.report is None

class TestAgentTrainingLoopDatasets:
    
    def test_build_preferences(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        dataset = loop.build_preferences(
            positive=lambda e: e.success,
            negative=lambda e: not e.success,
        )
        
        assert "prompt" in dataset.column_names
        assert "chosen" in dataset.column_names
        assert "rejected" in dataset.column_names
        assert len(dataset) == 6
    
    def test_build_preferences_custom_keys(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        dataset = loop.build_preferences(
            positive=lambda e: e.success,
            negative=lambda e: not e.success,
            prompt_key="input",
            chosen_key="preferred",
            rejected_key="dispreferred",
        )
        
        assert "input" in dataset.column_names
        assert "preferred" in dataset.column_names
        assert "dispreferred" in dataset.column_names
    
    def test_build_kto_dataset(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        dataset = loop.build_kto_dataset(
            desirable=lambda e: e.success,
        )
        
        assert "prompt" in dataset.column_names
        assert "completion" in dataset.column_names
        assert "label" in dataset.column_names
        assert len(dataset) == 5
    
    def test_build_sft_dataset(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        dataset = loop.build_sft_dataset(
            include=lambda e: e.success,
        )
        
        assert "text" in dataset.column_names
        assert len(dataset) == 3
    
    def test_build_sft_dataset_default_filter(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        dataset = loop.build_sft_dataset()
        
        assert len(dataset) == 3
    
    def test_build_grpo_dataset(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        dataset = loop.build_grpo_dataset()
        
        assert "prompt" in dataset.column_names
        assert "completion" in dataset.column_names
        assert "reward" in dataset.column_names
        assert len(dataset) == 5
    
    def test_build_grpo_dataset_custom_reward(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        dataset = loop.build_grpo_dataset(
            reward_fn=lambda e: 1.0 if e.success else -1.0,
        )
        
        rewards = dataset["reward"]
        assert sum(1 for r in rewards if r == 1.0) == 3
        assert sum(1 for r in rewards if r == -1.0) == 2

class TestAgentTrainingLoopFiltering:
    
    def test_filter(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        short = loop.filter(lambda e: e.total_steps < 10)
        
        assert len(short) == 3
        assert isinstance(short, AgentTrainingLoop)
    
    def test_successful(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        success = loop.successful()
        
        assert len(success) == 3
        assert success.success_rate == 1.0
    
    def test_failed(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        failed = loop.failed()
        
        assert len(failed) == 2
        assert failed.success_rate == 0.0
    
    def test_chained_filters(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        short_success = loop.successful().filter(lambda e: e.total_steps < 6)
        
        assert len(short_success) == 2

class TestAgentTrainingLoopComparison:
    
    def test_compare_basic(self):
        before_episodes = [
            make_episode("ep_001", success=False, steps=15),
            make_episode("ep_002", success=False, steps=20),
            make_episode("ep_003", success=True, steps=10),
        ]
        after_episodes = [
            make_episode("ep_004", success=True, steps=5),
            make_episode("ep_005", success=True, steps=8),
            make_episode("ep_006", success=True, steps=6),
        ]
        
        before = AgentTrainingLoop.from_episodes(before_episodes)
        after = AgentTrainingLoop.from_episodes(after_episodes)
        
        result = before.compare(after)
        
        assert result.before_success_rate == pytest.approx(1/3)
        assert result.after_success_rate == 1.0
        assert result.success_rate_delta > 0
        assert result.improved
    
    def test_compare_regression(self):
        before_episodes = [
            make_episode("ep_001", success=True, steps=5),
            make_episode("ep_002", success=True, steps=5),
        ]
        after_episodes = [
            make_episode("ep_003", success=False, steps=20),
            make_episode("ep_004", success=False, steps=25),
        ]
        
        before = AgentTrainingLoop.from_episodes(before_episodes)
        after = AgentTrainingLoop.from_episodes(after_episodes)
        
        result = before.compare(after)
        
        assert result.success_rate_delta < 0
        assert not result.improved
    
    def test_compare_no_change(self):
        episodes1 = [make_episode("ep_001", success=True, steps=10)]
        episodes2 = [make_episode("ep_002", success=True, steps=10)]
        
        loop1 = AgentTrainingLoop.from_episodes(episodes1)
        loop2 = AgentTrainingLoop.from_episodes(episodes2)
        
        result = loop1.compare(loop2)
        
        assert result.success_rate_delta == 0.0
        assert result.avg_steps_delta == 0.0
    
    def test_comparison_str_contains_sections(self):
        before = AgentTrainingLoop.from_episodes([make_episode("ep_001", True, 10)])
        after = AgentTrainingLoop.from_episodes([make_episode("ep_002", True, 5)])
        
        result = before.compare(after)
        output = str(result)
        
        assert "BEFORE/AFTER" in output
        assert "Success Rate" in output
        assert "Avg Steps" in output
        assert "Tool Error Rate" in output
    
    def test_comparison_str_shows_improvements(self):
        before = AgentTrainingLoop.from_episodes([make_episode("ep_001", False, 20)])
        after = AgentTrainingLoop.from_episodes([make_episode("ep_002", True, 5)])
        
        result = before.compare(after)
        output = str(result)
        
        assert "Improvements" in output
    
    def test_comparison_str_shows_regressions(self):
        before = AgentTrainingLoop.from_episodes([make_episode("ep_001", True, 5)])
        after = AgentTrainingLoop.from_episodes([make_episode("ep_002", False, 20)])
        
        result = before.compare(after)
        output = str(result)
        
        assert "Regressions" in output

class TestAgentTrainingLoopStats:
    
    def test_success_rate(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        assert loop.success_rate == 0.6
    
    def test_avg_steps(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        expected = (5 + 8 + 15 + 20 + 3) / 5
        assert loop.avg_steps == expected
    
    def test_tool_error_rate_no_errors(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        assert loop.tool_error_rate == 0.0
    
    def test_summary(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        summary = loop.summary()
        
        assert "5 episodes" in summary
        assert "60.0%" in summary
    
    def test_repr(self, sample_runs):
        loop = AgentTrainingLoop.from_runs(sample_runs)
        r = repr(loop)
        
        assert "AgentTrainingLoop" in r
        assert "episodes=5" in r
        assert "60.0%" in r

class TestComparisonResult:
    
    def test_improved_when_success_rate_up(self):
        result = ComparisonResult(
            before_success_rate=0.5,
            after_success_rate=0.8,
            success_rate_delta=0.3,
            before_avg_steps=10,
            after_avg_steps=10,
            avg_steps_delta=0,
            before_tool_error_rate=0.1,
            after_tool_error_rate=0.1,
            tool_error_rate_delta=0,
            before_episodes=100,
            after_episodes=100,
        )
        assert result.improved
    
    def test_improved_when_steps_down(self):
        result = ComparisonResult(
            before_success_rate=0.8,
            after_success_rate=0.8,
            success_rate_delta=0,
            before_avg_steps=20,
            after_avg_steps=10,
            avg_steps_delta=-10,
            before_tool_error_rate=0.1,
            after_tool_error_rate=0.1,
            tool_error_rate_delta=0,
            before_episodes=100,
            after_episodes=100,
        )
        assert result.improved
    
    def test_not_improved_when_all_regressed(self):
        result = ComparisonResult(
            before_success_rate=0.8,
            after_success_rate=0.5,
            success_rate_delta=-0.3,
            before_avg_steps=10,
            after_avg_steps=20,
            avg_steps_delta=10,
            before_tool_error_rate=0.05,
            after_tool_error_rate=0.2,
            tool_error_rate_delta=0.15,
            before_episodes=100,
            after_episodes=100,
        )
        assert not result.improved
