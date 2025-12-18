"""Agent training loop for outcome-driven post-training.

Provides a high-level API for the agent trace → TRL dataset workflow:
1. Load traces
2. Diagnose trace quality  
3. Build preference datasets
4. Compare before/after training

Example:
    from post_training_toolkit.agents import AgentTrainingLoop
    
    # Load and analyze
    loop = AgentTrainingLoop.from_traces("agent_runs.jsonl")
    print(loop.diagnose())
    
    # Build DPO dataset
    dataset = loop.build_preferences(
        positive=lambda e: e.success and e.total_steps < 15,
        negative=lambda e: not e.success,
    )
    
    # Train with TRL...
    
    # Compare before/after
    after = AgentTrainingLoop.from_traces("agent_runs_after.jsonl")
    print(loop.compare(after))
"""
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
    """Result of comparing two agent run logs (before/after training)."""
    
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
    
    # Optional metrics (may be None)
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
        
        # Summary
        improvements = []
        regressions = []
        
        if self.success_rate_delta > 0.01:
            improvements.append(f"success rate +{self.success_rate_delta:.1%}")
        elif self.success_rate_delta < -0.01:
            regressions.append(f"success rate {self.success_rate_delta:.1%}")
            
        if self.avg_steps_delta < -0.5:  # Fewer steps is better
            improvements.append(f"avg steps {self.avg_steps_delta:.1f}")
        elif self.avg_steps_delta > 0.5:
            regressions.append(f"avg steps +{self.avg_steps_delta:.1f}")
            
        if self.tool_error_rate_delta < -0.01:  # Fewer errors is better
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
        """Whether overall metrics improved."""
        # Weight success rate most heavily
        score = (
            (self.success_rate_delta * 2) +  # Success rate matters most
            (-self.avg_steps_delta / 10) +   # Fewer steps is good
            (-self.tool_error_rate_delta)    # Fewer errors is good
        )
        return score > 0


class AgentTrainingLoop:
    """High-level API for agent trace → training dataset workflow.
    
    This class provides a convenient interface for the outcome-driven
    agent post-training workflow:
    
    1. Load agent traces from JSONL files or AgentRunLog objects
    2. Run diagnostics to assess trace quality
    3. Build preference datasets (DPO) or labeled datasets (KTO/SFT)
    4. Compare before/after metrics to measure training impact
    
    Example:
        # Basic workflow
        loop = AgentTrainingLoop.from_traces("runs.jsonl")
        
        # Check trace quality
        report = loop.diagnose()
        if report.has_critical_issues:
            print("Warning:", report)
        
        # Build DPO dataset
        dataset = loop.build_preferences(
            positive=lambda e: e.success and e.total_steps < 15,
            negative=lambda e: not e.success,
        )
        
        # After training, compare
        after_loop = AgentTrainingLoop.from_traces("runs_after.jsonl")
        comparison = loop.compare(after_loop)
        print(comparison)
    """
    
    def __init__(self, runs: AgentRunLog):
        """Initialize with an AgentRunLog.
        
        Use from_traces() or from_runs() classmethods instead of direct init.
        """
        self._runs = runs
        self._report: Optional[AgentDiagnosticsReport] = None
    
    @classmethod
    def from_traces(cls, path: Union[str, Path]) -> "AgentTrainingLoop":
        """Load from a JSONL trace file.
        
        Args:
            path: Path to JSONL file with agent traces
            
        Returns:
            AgentTrainingLoop instance
        """
        runs = AgentRunLog.from_jsonl(path)
        return cls(runs)
    
    @classmethod
    def from_runs(cls, runs: AgentRunLog) -> "AgentTrainingLoop":
        """Create from an existing AgentRunLog.
        
        Args:
            runs: Pre-loaded AgentRunLog
            
        Returns:
            AgentTrainingLoop instance
        """
        return cls(runs)
    
    @classmethod
    def from_episodes(cls, episodes: List[Episode]) -> "AgentTrainingLoop":
        """Create from a list of Episode objects.
        
        Args:
            episodes: List of Episode objects
            
        Returns:
            AgentTrainingLoop instance
        """
        runs = AgentRunLog.from_episodes(episodes)
        return cls(runs)
    
    @property
    def runs(self) -> AgentRunLog:
        """Access the underlying AgentRunLog."""
        return self._runs
    
    @property
    def episodes(self) -> List[Episode]:
        """Access all episodes."""
        return self._runs.episodes
    
    def __len__(self) -> int:
        """Number of episodes."""
        return len(self._runs)
    
    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    
    def diagnose(self, budget_per_episode: Optional[float] = None) -> AgentDiagnosticsReport:
        """Run diagnostics on the traces.
        
        Analyzes traces for common issues like low success rate,
        tool errors, looping behavior, token/cost anomalies.
        
        Args:
            budget_per_episode: Optional cost budget per episode
            
        Returns:
            AgentDiagnosticsReport with metrics and insights
        """
        self._report = analyze_runs(self._runs, budget_per_episode=budget_per_episode)
        return self._report
    
    @property
    def report(self) -> Optional[AgentDiagnosticsReport]:
        """Get cached diagnostics report (None if diagnose() not called)."""
        return self._report
    
    # -------------------------------------------------------------------------
    # Dataset Building
    # -------------------------------------------------------------------------
    
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
        """Build a DPO preference dataset from traces.
        
        Creates (chosen, rejected) pairs based on outcome predicates.
        
        Common patterns:
            # Success vs failure
            positive=lambda e: e.success,
            negative=lambda e: not e.success,
            
            # Efficient success vs inefficient
            positive=lambda e: e.success and e.total_steps < 10,
            negative=lambda e: e.success and e.total_steps > 20,
            
            # No loops vs loops
            positive=lambda e: e.success and not e.has_repeated_tool_pattern(),
            negative=lambda e: e.has_repeated_tool_pattern(),
        
        Args:
            positive: Predicate for "chosen" episodes (good behavior)
            negative: Predicate for "rejected" episodes (bad behavior)
            prompt_key: Key for prompt in output dataset
            chosen_key: Key for chosen completion in output dataset
            rejected_key: Key for rejected completion in output dataset
            format_fn: Custom episode formatter (default: conversation format)
            require_same_prompt: Only pair episodes with identical prompts
            
        Returns:
            HuggingFace Dataset with columns [prompt, chosen, rejected]
        """
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
        """Build a KTO dataset from traces.
        
        KTO uses binary labels (desirable/undesirable) per example.
        
        Args:
            desirable: Predicate for desirable episodes (label=True)
            prompt_key: Key for prompt in output dataset
            completion_key: Key for completion in output dataset
            label_key: Key for binary label in output dataset
            format_fn: Custom episode formatter
            
        Returns:
            HuggingFace Dataset with columns [prompt, completion, label]
        """
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
        """Build an SFT dataset from traces.
        
        Args:
            include: Predicate to filter episodes (default: successful only)
            text_key: Key for text in output dataset
            format_fn: Custom episode formatter
            
        Returns:
            HuggingFace Dataset with column [text]
        """
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
        """Build a GRPO dataset from traces.
        
        GRPO uses reward signals directly rather than preferences.
        
        Args:
            reward_fn: Function to compute reward for episode
                      (default: episode.reward or success-based)
            prompt_key: Key for prompt in output dataset
            completion_key: Key for completion in output dataset
            reward_key: Key for reward in output dataset
            format_fn: Custom episode formatter
            
        Returns:
            HuggingFace Dataset with columns [prompt, completion, reward]
        """
        return to_grpo_dataset(
            self._runs,
            reward_fn=reward_fn,
            prompt_key=prompt_key,
            completion_key=completion_key,
            reward_key=reward_key,
            format_fn=format_fn,
        )
    
    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------
    
    def filter(self, predicate: Callable[[Episode], bool]) -> "AgentTrainingLoop":
        """Filter episodes, returning a new AgentTrainingLoop.
        
        Args:
            predicate: Function that returns True for episodes to keep
            
        Returns:
            New AgentTrainingLoop with filtered episodes
        """
        filtered_runs = self._runs.filter(predicate)
        return AgentTrainingLoop(filtered_runs)
    
    def successful(self) -> "AgentTrainingLoop":
        """Get only successful episodes."""
        return self.filter(lambda e: e.success is True)
    
    def failed(self) -> "AgentTrainingLoop":
        """Get only failed episodes."""
        return self.filter(lambda e: e.success is False)
    
    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    
    def compare(self, other: "AgentTrainingLoop") -> ComparisonResult:
        """Compare metrics between this loop and another (e.g., after training).
        
        Args:
            other: Another AgentTrainingLoop to compare against
            
        Returns:
            ComparisonResult with before/after metrics and deltas
        """
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
    
    # -------------------------------------------------------------------------
    # Stats (convenience access)
    # -------------------------------------------------------------------------
    
    @property
    def success_rate(self) -> float:
        """Success rate across all episodes."""
        return self._runs.success_rate
    
    @property
    def avg_steps(self) -> float:
        """Average steps per episode."""
        return self._runs.avg_steps
    
    @property
    def tool_error_rate(self) -> float:
        """Overall tool error rate."""
        return self._runs.tool_error_rate
    
    def summary(self) -> str:
        """Get a brief summary string."""
        return (
            f"AgentTrainingLoop: {len(self)} episodes, "
            f"{self.success_rate:.1%} success rate, "
            f"{self.avg_steps:.1f} avg steps"
        )
    
    def __repr__(self) -> str:
        return f"AgentTrainingLoop(episodes={len(self)}, success_rate={self.success_rate:.1%})"
