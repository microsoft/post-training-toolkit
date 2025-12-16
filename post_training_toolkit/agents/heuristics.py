"""Agent-specific diagnostics and heuristics.

Analyzes agent run logs to detect common failure modes:
- Low success rate
- Tool error spikes
- Looping behavior (repeated tool patterns)
- Token/cost runaway
- Step count anomalies

Example:
    from post_training_toolkit.agents import AgentRunLog, analyze_runs
    
    runs = AgentRunLog.from_jsonl("agent_runs.jsonl")
    report = analyze_runs(runs)
    print(report)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import statistics

from post_training_toolkit.agents.traces import AgentRunLog, Episode


@dataclass
class AgentInsight:
    """A diagnostic insight from agent trace analysis.
    
    Attributes:
        type: Unique identifier (e.g., "low_success_rate")
        severity: "high", "medium", or "low"
        message: Human-readable description
        episodes: Episode IDs where this issue was detected
        data: Additional diagnostic data
    """
    type: str
    severity: str  # "high", "medium", "low"
    message: str
    episodes: List[str] = field(default_factory=list)
    data: Dict = field(default_factory=dict)


@dataclass
class AgentDiagnosticsReport:
    """Full diagnostics report for agent runs.
    
    Attributes:
        total_episodes: Number of episodes analyzed
        success_rate: Overall success rate
        avg_steps: Average steps per episode
        avg_tokens: Average tokens per episode (if available)
        total_cost: Total cost (if available)
        tool_error_rate: Overall tool error rate
        insights: List of detected issues
    """
    total_episodes: int
    success_rate: float
    avg_steps: float
    avg_tokens: Optional[float]
    total_cost: Optional[float]
    tool_error_rate: float
    insights: List[AgentInsight] = field(default_factory=list)
    
    # Breakdown stats
    episodes_with_loops: int = 0
    episodes_with_tool_errors: int = 0
    step_distribution: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Format as human-readable report."""
        lines = [
            "=" * 60,
            "AGENT DIAGNOSTICS REPORT",
            "=" * 60,
            f"Episodes analyzed: {self.total_episodes}",
            "",
            "--- Summary Metrics ---",
            f"  Success rate: {self.success_rate:.1%}",
            f"  Avg steps per episode: {self.avg_steps:.1f}",
        ]
        
        if self.avg_tokens is not None:
            lines.append(f"  Avg tokens per episode: {self.avg_tokens:.0f}")
        if self.total_cost is not None:
            lines.append(f"  Total cost: ${self.total_cost:.4f}")
        
        lines.extend([
            f"  Tool error rate: {self.tool_error_rate:.1%}",
            f"  Episodes with loops: {self.episodes_with_loops}",
            f"  Episodes with tool errors: {self.episodes_with_tool_errors}",
        ])
        
        if self.step_distribution:
            lines.append("")
            lines.append("--- Step Distribution ---")
            for k, v in self.step_distribution.items():
                lines.append(f"  {k}: {v:.1f}")
        
        if self.insights:
            lines.append("")
            lines.append("--- Insights ---")
            
            high = [i for i in self.insights if i.severity == "high"]
            medium = [i for i in self.insights if i.severity == "medium"]
            low = [i for i in self.insights if i.severity == "low"]
            
            if high:
                lines.append(f"  🚨 HIGH SEVERITY ({len(high)}):")
                for insight in high:
                    lines.append(f"     • {insight.message}")
            
            if medium:
                lines.append(f"  ⚠️  MEDIUM SEVERITY ({len(medium)}):")
                for insight in medium:
                    lines.append(f"     • {insight.message}")
            
            if low:
                lines.append(f"  ℹ️  LOW SEVERITY ({len(low)}):")
                for insight in low:
                    lines.append(f"     • {insight.message}")
        else:
            lines.append("")
            lines.append("--- Insights ---")
            lines.append("  ✓ No issues detected")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    @property
    def has_critical_issues(self) -> bool:
        """Whether any high-severity issues were detected."""
        return any(i.severity == "high" for i in self.insights)


def detect_low_success_rate(
    runs: AgentRunLog,
    threshold: float = 0.5,
    min_episodes: int = 5,
) -> List[AgentInsight]:
    """Detect if success rate is below threshold."""
    if len(runs) < min_episodes:
        return []
    
    rate = runs.success_rate
    if rate < threshold:
        severity = "high" if rate < 0.3 else "medium"
        failed = [e.episode_id for e in runs if e.success is False]
        return [AgentInsight(
            type="low_success_rate",
            severity=severity,
            message=f"Success rate is {rate:.1%} (threshold: {threshold:.1%}). "
                    f"{len(failed)} episodes failed.",
            episodes=failed[:10],  # Cap at 10
            data={"success_rate": rate, "threshold": threshold, "failed_count": len(failed)},
        )]
    return []


def detect_tool_error_spikes(
    runs: AgentRunLog,
    threshold: float = 0.2,
) -> List[AgentInsight]:
    """Detect high tool error rates."""
    rate = runs.tool_error_rate
    if rate > threshold:
        severity = "high" if rate > 0.4 else "medium"
        affected = [e.episode_id for e in runs if e.tool_error_rate > threshold]
        return [AgentInsight(
            type="tool_error_spike",
            severity=severity,
            message=f"Tool error rate is {rate:.1%} (threshold: {threshold:.1%}). "
                    f"{len(affected)} episodes affected.",
            episodes=affected[:10],
            data={"tool_error_rate": rate, "threshold": threshold},
        )]
    return []


def detect_loops(
    runs: AgentRunLog,
    min_repeats: int = 3,
) -> List[AgentInsight]:
    """Detect episodes with repeated tool call patterns (loops)."""
    looping = []
    for episode in runs:
        if episode.has_repeated_tool_pattern(min_repeats=min_repeats):
            looping.append(episode.episode_id)
    
    if looping:
        severity = "high" if len(looping) > len(runs) * 0.2 else "medium"
        return [AgentInsight(
            type="loop_detected",
            severity=severity,
            message=f"Detected looping behavior in {len(looping)} episodes "
                    f"({len(looping)/len(runs):.1%} of runs).",
            episodes=looping[:10],
            data={"loop_count": len(looping), "min_repeats": min_repeats},
        )]
    return []


def detect_token_runaway(
    runs: AgentRunLog,
    threshold_multiplier: float = 3.0,
    min_episodes: int = 5,
) -> List[AgentInsight]:
    """Detect episodes with unusually high token usage."""
    tokens = [e.total_tokens for e in runs if e.total_tokens is not None]
    if len(tokens) < min_episodes:
        return []
    
    median = statistics.median(tokens)
    threshold = median * threshold_multiplier
    
    runaway = [
        e.episode_id for e in runs 
        if e.total_tokens is not None and e.total_tokens > threshold
    ]
    
    if runaway:
        max_tokens = max(e.total_tokens for e in runs if e.total_tokens is not None)
        return [AgentInsight(
            type="token_runaway",
            severity="high" if len(runaway) > 3 else "medium",
            message=f"{len(runaway)} episodes exceeded {threshold_multiplier}x median token usage "
                    f"(median: {median:.0f}, max: {max_tokens:.0f}).",
            episodes=runaway[:10],
            data={
                "median_tokens": median, 
                "threshold": threshold, 
                "max_tokens": max_tokens,
                "runaway_count": len(runaway),
            },
        )]
    return []


def detect_step_anomalies(
    runs: AgentRunLog,
    threshold_multiplier: float = 3.0,
    min_episodes: int = 5,
) -> List[AgentInsight]:
    """Detect episodes with unusually high step counts."""
    if len(runs) < min_episodes:
        return []
    
    steps = [e.total_steps for e in runs]
    median = statistics.median(steps)
    threshold = max(median * threshold_multiplier, 20)  # At least 20 steps
    
    long_episodes = [
        e.episode_id for e in runs 
        if e.total_steps > threshold
    ]
    
    if long_episodes:
        max_steps = max(e.total_steps for e in runs)
        return [AgentInsight(
            type="step_anomaly",
            severity="medium",
            message=f"{len(long_episodes)} episodes had unusually high step counts "
                    f"(median: {median:.0f}, max: {max_steps}).",
            episodes=long_episodes[:10],
            data={
                "median_steps": median,
                "threshold": threshold,
                "max_steps": max_steps,
            },
        )]
    return []


def detect_cost_anomalies(
    runs: AgentRunLog,
    budget_per_episode: Optional[float] = None,
    threshold_multiplier: float = 3.0,
) -> List[AgentInsight]:
    """Detect episodes that exceeded cost budgets."""
    costs = [e.total_cost for e in runs if e.total_cost is not None]
    if not costs:
        return []
    
    if budget_per_episode is not None:
        over_budget = [
            e.episode_id for e in runs
            if e.total_cost is not None and e.total_cost > budget_per_episode
        ]
        if over_budget:
            total_over = sum(
                e.total_cost - budget_per_episode 
                for e in runs 
                if e.total_cost is not None and e.total_cost > budget_per_episode
            )
            return [AgentInsight(
                type="budget_exceeded",
                severity="high",
                message=f"{len(over_budget)} episodes exceeded budget of ${budget_per_episode:.4f} "
                        f"(total overage: ${total_over:.4f}).",
                episodes=over_budget[:10],
                data={"budget": budget_per_episode, "overage": total_over},
            )]
    
    # Check for outliers even without explicit budget
    median = statistics.median(costs)
    threshold = median * threshold_multiplier
    expensive = [
        e.episode_id for e in runs
        if e.total_cost is not None and e.total_cost > threshold
    ]
    
    if expensive and len(expensive) < len(runs) * 0.2:  # Only flag if it's outliers
        return [AgentInsight(
            type="cost_outliers",
            severity="medium",
            message=f"{len(expensive)} episodes had costs >{threshold_multiplier}x median "
                    f"(median: ${median:.4f}).",
            episodes=expensive[:10],
            data={"median_cost": median, "threshold": threshold},
        )]
    
    return []


def analyze_runs(
    runs: AgentRunLog,
    budget_per_episode: Optional[float] = None,
) -> AgentDiagnosticsReport:
    """Run all agent diagnostics and generate a report.
    
    Args:
        runs: Agent run log to analyze
        budget_per_episode: Optional cost budget per episode
        
    Returns:
        AgentDiagnosticsReport with metrics and insights
    """
    insights: List[AgentInsight] = []
    
    # Run all heuristics
    insights.extend(detect_low_success_rate(runs))
    insights.extend(detect_tool_error_spikes(runs))
    insights.extend(detect_loops(runs))
    insights.extend(detect_token_runaway(runs))
    insights.extend(detect_step_anomalies(runs))
    insights.extend(detect_cost_anomalies(runs, budget_per_episode=budget_per_episode))
    
    # Compute additional stats
    episodes_with_loops = sum(1 for e in runs if e.has_repeated_tool_pattern())
    episodes_with_tool_errors = sum(1 for e in runs if e.tool_errors)
    
    # Step distribution
    steps = [e.total_steps for e in runs]
    step_distribution = {}
    if steps:
        step_distribution = {
            "min": min(steps),
            "max": max(steps),
            "median": statistics.median(steps),
            "mean": statistics.mean(steps),
        }
        if len(steps) > 1:
            step_distribution["std"] = statistics.stdev(steps)
    
    return AgentDiagnosticsReport(
        total_episodes=len(runs),
        success_rate=runs.success_rate,
        avg_steps=runs.avg_steps,
        avg_tokens=runs.avg_tokens,
        total_cost=runs.total_cost,
        tool_error_rate=runs.tool_error_rate,
        insights=insights,
        episodes_with_loops=episodes_with_loops,
        episodes_with_tool_errors=episodes_with_tool_errors,
        step_distribution=step_distribution,
    )
