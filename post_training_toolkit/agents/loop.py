from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

from post_training_toolkit.agents.traces import AgentRunLog, Episode
from post_training_toolkit.agents.heuristics import (
    AgentDiagnosticsReport,
    analyze_runs,
)
from post_training_toolkit.agents.converters import (
    to_preference_pairs,
    to_kto_dataset,
    to_sft_dataset,
    to_grpo_dataset,
    format_episode_as_conversation,
)

@dataclass
class ComparisonResult:
    
    before_success_rate: float
    after_success_rate: float
    success_rate_delta: float
    
    before_avg_steps: float
    after_avg_steps: float
    avg_steps_delta: float
    
    before_tool_error_rate: float
    after_tool_error_rate: float
    tool_error_rate_delta: float
    
    before_episodes: int
    after_episodes: int
    
    before_avg_tokens: Optional[float] = None
    after_avg_tokens: Optional[float] = None
    avg_tokens_delta: Optional[float] = None
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "BEFORE/AFTER COMPARISON",
            "=" * 50,
            "",
            f"Episodes:        {self.before_episodes} → {self.after_episodes}",
            "",
            "--- Success Rate ---",
            f"  Before: {self.before_success_rate:.1%}",
            f"  After:  {self.after_success_rate:.1%}",
            f"  Delta:  {self.success_rate_delta:+.1%}",
            "",
            "--- Avg Steps ---",
            f"  Before: {self.before_avg_steps:.1f}",
            f"  After:  {self.after_avg_steps:.1f}",
            f"  Delta:  {self.avg_steps_delta:+.1f}",
            "",
            "--- Tool Error Rate ---",
            f"  Before: {self.before_tool_error_rate:.1%}",
            f"  After:  {self.after_tool_error_rate:.1%}",
            f"  Delta:  {self.tool_error_rate_delta:+.1%}",
        ]
        
        if self.before_avg_tokens is not None and self.after_avg_tokens is not None:
            lines.extend([
                "",
                "--- Avg Tokens ---",
                f"  Before: {self.before_avg_tokens:.0f}",
                f"  After:  {self.after_avg_tokens:.0f}",
                f"  Delta:  {self.avg_tokens_delta:+.0f}",
            ])
        
        lines.append("=" * 50)
        
        improvements = []
        regressions = []
        
        if self.success_rate_delta > 0.01:
            improvements.append(f"success rate +{self.success_rate_delta:.1%}")
        elif self.success_rate_delta < -0.01:
            regressions.append(f"success rate {self.success_rate_delta:.1%}")
            
        if self.avg_steps_delta < -0.5:
            improvements.append(f"avg steps {self.avg_steps_delta:.1f}")
        elif self.avg_steps_delta > 0.5:
            regressions.append(f"avg steps +{self.avg_steps_delta:.1f}")
            
        if self.tool_error_rate_delta < -0.01:
            improvements.append(f"tool errors {self.tool_error_rate_delta:.1%}")
        elif self.tool_error_rate_delta > 0.01:
            regressions.append(f"tool errors +{self.tool_error_rate_delta:.1%}")
        
        if improvements:
            lines.append(f"✓ Improvements: {', '.join(improvements)}")
        if regressions:
            lines.append(f"⚠ Regressions: {', '.join(regressions)}")
        if not improvements and not regressions:
            lines.append("→ No significant changes")
        
        return "\n".join(lines)
    
    @property
    def improved(self) -> bool:
        score = (
            (self.success_rate_delta * 2) +
            (-self.avg_steps_delta / 10) +
            (-self.tool_error_rate_delta)
        )
        return score > 0

class AgentTrainingLoop:
    
    def __init__(self, runs: AgentRunLog):
        self._runs = runs
        self._report: Optional[AgentDiagnosticsReport] = None
    
    @classmethod
    def from_traces(cls, path: Union[str, Path]) -> "AgentTrainingLoop":
        runs = AgentRunLog.from_jsonl(path)
        return cls(runs)
    
    @classmethod
    def from_runs(cls, runs: AgentRunLog) -> "AgentTrainingLoop":
        return cls(runs)
    
    @classmethod
    def from_episodes(cls, episodes: List[Episode]) -> "AgentTrainingLoop":
        runs = AgentRunLog.from_episodes(episodes)
        return cls(runs)
    
    @property
    def runs(self) -> AgentRunLog:
        return self._runs
    
    @property
    def episodes(self) -> List[Episode]:
        return self._runs.episodes
    
    def __len__(self) -> int:
        return len(self._runs)
    
    
    def diagnose(self, budget_per_episode: Optional[float] = None) -> AgentDiagnosticsReport:
        self._report = analyze_runs(self._runs, budget_per_episode=budget_per_episode)
        return self._report
    
    @property
    def report(self) -> Optional[AgentDiagnosticsReport]:
        return self._report
    
    
    def build_preferences(
        self,
        positive: Callable[[Episode], bool],
        negative: Callable[[Episode], bool],
        *,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        format_fn: Optional[Callable[[Episode], str]] = None,
        require_same_prompt: bool = False,
    ) -> "Dataset":
        return to_preference_pairs(
            self._runs,
            positive=positive,
            negative=negative,
            prompt_key=prompt_key,
            chosen_key=chosen_key,
            rejected_key=rejected_key,
            format_fn=format_fn,
            require_same_prompt=require_same_prompt,
        )
    
    def build_kto_dataset(
        self,
        desirable: Callable[[Episode], bool],
        *,
        prompt_key: str = "prompt",
        completion_key: str = "completion",
        label_key: str = "label",
        format_fn: Optional[Callable[[Episode], str]] = None,
    ) -> "Dataset":
        return to_kto_dataset(
            self._runs,
            desirable=desirable,
            prompt_key=prompt_key,
            completion_key=completion_key,
            label_key=label_key,
            format_fn=format_fn,
        )
    
    def build_sft_dataset(
        self,
        include: Optional[Callable[[Episode], bool]] = None,
        *,
        text_key: str = "text",
        format_fn: Optional[Callable[[Episode], str]] = None,
    ) -> "Dataset":
        return to_sft_dataset(
            self._runs,
            include=include,
            text_key=text_key,
            format_fn=format_fn,
        )
    
    def build_grpo_dataset(
        self,
        reward_fn: Optional[Callable[[Episode], float]] = None,
        *,
        prompt_key: str = "prompt",
        completion_key: str = "completion",
        reward_key: str = "reward",
        format_fn: Optional[Callable[[Episode], str]] = None,
    ) -> "Dataset":
        return to_grpo_dataset(
            self._runs,
            reward_fn=reward_fn,
            prompt_key=prompt_key,
            completion_key=completion_key,
            reward_key=reward_key,
            format_fn=format_fn,
        )
    
    
    def filter(self, predicate: Callable[[Episode], bool]) -> "AgentTrainingLoop":
        filtered_runs = self._runs.filter(predicate)
        return AgentTrainingLoop(filtered_runs)
    
    def successful(self) -> "AgentTrainingLoop":
        return self.filter(lambda e: e.success is True)
    
    def failed(self) -> "AgentTrainingLoop":
        return self.filter(lambda e: e.success is False)
    
    
    def compare(self, other: "AgentTrainingLoop") -> ComparisonResult:
        before = self._runs
        after = other._runs
        
        before_tokens = before.avg_tokens
        after_tokens = after.avg_tokens
        tokens_delta = None
        if before_tokens is not None and after_tokens is not None:
            tokens_delta = after_tokens - before_tokens
        
        return ComparisonResult(
            before_success_rate=before.success_rate,
            after_success_rate=after.success_rate,
            success_rate_delta=after.success_rate - before.success_rate,
            
            before_avg_steps=before.avg_steps,
            after_avg_steps=after.avg_steps,
            avg_steps_delta=after.avg_steps - before.avg_steps,
            
            before_tool_error_rate=before.tool_error_rate,
            after_tool_error_rate=after.tool_error_rate,
            tool_error_rate_delta=after.tool_error_rate - before.tool_error_rate,
            
            before_episodes=len(before),
            after_episodes=len(after),
            
            before_avg_tokens=before_tokens,
            after_avg_tokens=after_tokens,
            avg_tokens_delta=tokens_delta,
        )
    
    
    @property
    def success_rate(self) -> float:
        return self._runs.success_rate
    
    @property
    def avg_steps(self) -> float:
        return self._runs.avg_steps
    
    @property
    def tool_error_rate(self) -> float:
        return self._runs.tool_error_rate
    
    def summary(self) -> str:
        return (
            f"AgentTrainingLoop: {len(self)} episodes, "
            f"{self.success_rate:.1%} success rate, "
            f"{self.avg_steps:.1f} avg steps"
        )
    
    def __repr__(self) -> str:
        return f"AgentTrainingLoop(episodes={len(self)}, success_rate={self.success_rate:.1%})"
