"""Agent trace schema and loading utilities.

Defines the canonical PTT Trace v1 format for agent runs.
Users can either log directly in this format or write a simple adapter.

Trace Format (JSONL):
    Each line is a step event within an episode:
    
    {"episode_id": "ep_001", "step": 0, "type": "user_message", "content": "Find flights to Paris"}
    {"episode_id": "ep_001", "step": 1, "type": "assistant_message", "content": "I'll search for flights..."}
    {"episode_id": "ep_001", "step": 2, "type": "tool_call", "tool": "search_flights", "args": {"dest": "Paris"}}
    {"episode_id": "ep_001", "step": 3, "type": "tool_result", "tool": "search_flights", "result": "...", "error": null}
    {"episode_id": "ep_001", "step": 4, "type": "assistant_message", "content": "I found 3 flights..."}
    {"episode_id": "ep_001", "step": 5, "type": "episode_end", "success": true, "reward": 1.0, "total_tokens": 1523}

Minimal Required Fields:
    - episode_id: str — unique identifier for the episode
    - step: int — step number within episode (0-indexed)
    - type: str — one of: user_message, assistant_message, tool_call, tool_result, episode_end

Optional Fields:
    - content: str — message content (for message types)
    - tool: str — tool name (for tool_call/tool_result)
    - args: dict — tool arguments (for tool_call)
    - result: str — tool output (for tool_result)
    - error: str|null — error message if tool failed
    - success: bool — whether episode succeeded (for episode_end)
    - reward: float — reward signal (for episode_end)
    - total_tokens: int — tokens used in episode
    - total_cost: float — cost in dollars
    - timestamp: str — ISO timestamp
    - metadata: dict — any additional data
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union


class StepType(str, Enum):
    """Types of steps in an agent trace."""
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    EPISODE_END = "episode_end"
    # Allow unknown types for flexibility
    OTHER = "other"
    
    @classmethod
    def from_str(cls, s: str) -> "StepType":
        """Convert string to StepType, defaulting to OTHER for unknown."""
        try:
            return cls(s)
        except ValueError:
            return cls.OTHER


@dataclass
class Step:
    """A single step in an agent episode.
    
    Attributes:
        episode_id: Unique identifier for the episode
        step: Step number within episode (0-indexed)
        type: Type of step (message, tool call, etc.)
        content: Message content (for message types)
        tool: Tool name (for tool_call/tool_result)
        args: Tool arguments (for tool_call)
        result: Tool output (for tool_result)
        error: Error message if step failed
        timestamp: When this step occurred
        tokens: Tokens used in this step
        metadata: Additional step-level data
    """
    episode_id: str
    step: int
    type: StepType
    content: Optional[str] = None
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None
    tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step":
        """Create Step from dictionary."""
        return cls(
            episode_id=d.get("episode_id", ""),
            step=d.get("step", 0),
            type=StepType.from_str(d.get("type", "other")),
            content=d.get("content"),
            tool=d.get("tool"),
            args=d.get("args"),
            result=d.get("result"),
            error=d.get("error"),
            timestamp=d.get("timestamp"),
            tokens=d.get("tokens"),
            metadata=d.get("metadata", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "episode_id": self.episode_id,
            "step": self.step,
            "type": self.type.value,
        }
        if self.content is not None:
            d["content"] = self.content
        if self.tool is not None:
            d["tool"] = self.tool
        if self.args is not None:
            d["args"] = self.args
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp
        if self.tokens is not None:
            d["tokens"] = self.tokens
        if self.metadata:
            d["metadata"] = self.metadata
        return d
    
    @property
    def is_tool_error(self) -> bool:
        """Whether this step is a tool result with an error."""
        return self.type == StepType.TOOL_RESULT and self.error is not None


@dataclass
class Episode:
    """A complete agent episode (one task attempt).
    
    Attributes:
        episode_id: Unique identifier
        steps: List of steps in order
        success: Whether the episode succeeded (from episode_end)
        reward: Reward signal (from episode_end)
        total_tokens: Total tokens used
        total_cost: Total cost in dollars
        metadata: Episode-level metadata
    """
    episode_id: str
    steps: List[Step] = field(default_factory=list)
    success: Optional[bool] = None
    reward: Optional[float] = None
    total_tokens: Optional[int] = None
    total_cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_steps(self) -> int:
        """Number of steps in the episode."""
        return len(self.steps)
    
    @property
    def user_messages(self) -> List[Step]:
        """Get all user message steps."""
        return [s for s in self.steps if s.type == StepType.USER_MESSAGE]
    
    @property
    def assistant_messages(self) -> List[Step]:
        """Get all assistant message steps."""
        return [s for s in self.steps if s.type == StepType.ASSISTANT_MESSAGE]
    
    @property
    def tool_calls(self) -> List[Step]:
        """Get all tool call steps."""
        return [s for s in self.steps if s.type == StepType.TOOL_CALL]
    
    @property
    def tool_results(self) -> List[Step]:
        """Get all tool result steps."""
        return [s for s in self.steps if s.type == StepType.TOOL_RESULT]
    
    @property
    def tool_errors(self) -> List[Step]:
        """Get all tool results that had errors."""
        return [s for s in self.steps if s.is_tool_error]
    
    @property
    def tool_error_rate(self) -> float:
        """Fraction of tool calls that resulted in errors."""
        tool_results = self.tool_results
        if not tool_results:
            return 0.0
        return len(self.tool_errors) / len(tool_results)
    
    @property
    def initial_prompt(self) -> Optional[str]:
        """Get the initial user message (task prompt)."""
        user_msgs = self.user_messages
        return user_msgs[0].content if user_msgs else None
    
    @property
    def final_response(self) -> Optional[str]:
        """Get the final assistant message."""
        asst_msgs = self.assistant_messages
        return asst_msgs[-1].content if asst_msgs else None
    
    def get_tool_call_sequence(self) -> List[str]:
        """Get sequence of tool names called."""
        return [s.tool for s in self.tool_calls if s.tool]
    
    def has_repeated_tool_pattern(self, min_repeats: int = 3) -> bool:
        """Check if there's a repeated tool call pattern (loop detection)."""
        tools = self.get_tool_call_sequence()
        if len(tools) < min_repeats:
            return False
        
        # Check for simple repeats: same tool called N+ times in a row
        for i in range(len(tools) - min_repeats + 1):
            if len(set(tools[i:i + min_repeats])) == 1:
                return True
        
        # Check for pattern repeats: [A, B, A, B, A, B]
        for pattern_len in range(1, len(tools) // min_repeats + 1):
            pattern = tuple(tools[:pattern_len])
            repeats = 0
            for i in range(0, len(tools) - pattern_len + 1, pattern_len):
                if tuple(tools[i:i + pattern_len]) == pattern:
                    repeats += 1
            if repeats >= min_repeats:
                return True
        
        return False


class AgentRunLog:
    """Collection of agent episodes loaded from traces.
    
    This is the main entry point for loading and working with agent traces.
    
    Example:
        # Load from JSONL
        runs = AgentRunLog.from_jsonl("agent_runs.jsonl")
        
        # Filter episodes
        successful = runs.filter(lambda e: e.success)
        
        # Iterate
        for episode in runs:
            print(f"{episode.episode_id}: {episode.total_steps} steps")
    """
    
    def __init__(self, episodes: List[Episode]):
        """Initialize with list of episodes."""
        self._episodes = episodes
        self._by_id = {e.episode_id: e for e in episodes}
    
    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> "AgentRunLog":
        """Load traces from a JSONL file.
        
        Each line should be a step event with at least:
        - episode_id: str
        - step: int  
        - type: str
        
        Steps are grouped into Episodes automatically.
        """
        path = Path(path)
        steps_by_episode: Dict[str, List[Step]] = {}
        episode_metadata: Dict[str, Dict[str, Any]] = {}
        
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                step = Step.from_dict(obj)
                
                if step.episode_id not in steps_by_episode:
                    steps_by_episode[step.episode_id] = []
                steps_by_episode[step.episode_id].append(step)
                
                # Extract episode-level metadata from episode_end
                if step.type == StepType.EPISODE_END:
                    episode_metadata[step.episode_id] = {
                        "success": obj.get("success"),
                        "reward": obj.get("reward"),
                        "total_tokens": obj.get("total_tokens"),
                        "total_cost": obj.get("total_cost"),
                        "metadata": obj.get("metadata", {}),
                    }
        
        # Build episodes
        episodes = []
        for episode_id, steps in steps_by_episode.items():
            steps.sort(key=lambda s: s.step)
            meta = episode_metadata.get(episode_id, {})
            
            # Sum tokens from steps if not in metadata
            total_tokens = meta.get("total_tokens")
            if total_tokens is None:
                step_tokens = [s.tokens for s in steps if s.tokens is not None]
                total_tokens = sum(step_tokens) if step_tokens else None
            
            episode = Episode(
                episode_id=episode_id,
                steps=steps,
                success=meta.get("success"),
                reward=meta.get("reward"),
                total_tokens=total_tokens,
                total_cost=meta.get("total_cost"),
                metadata=meta.get("metadata", {}),
            )
            episodes.append(episode)
        
        # Sort by episode_id for determinism
        episodes.sort(key=lambda e: e.episode_id)
        return cls(episodes)
    
    @classmethod
    def from_episodes(cls, episodes: List[Episode]) -> "AgentRunLog":
        """Create from a list of Episode objects."""
        return cls(episodes)
    
    @classmethod
    def from_dicts(cls, records: List[Dict[str, Any]]) -> "AgentRunLog":
        """Create from a list of step dictionaries."""
        # Write to temp format and use from_jsonl logic
        steps_by_episode: Dict[str, List[Step]] = {}
        episode_metadata: Dict[str, Dict[str, Any]] = {}
        
        for obj in records:
            step = Step.from_dict(obj)
            if step.episode_id not in steps_by_episode:
                steps_by_episode[step.episode_id] = []
            steps_by_episode[step.episode_id].append(step)
            
            if step.type == StepType.EPISODE_END:
                episode_metadata[step.episode_id] = {
                    "success": obj.get("success"),
                    "reward": obj.get("reward"),
                    "total_tokens": obj.get("total_tokens"),
                    "total_cost": obj.get("total_cost"),
                    "metadata": obj.get("metadata", {}),
                }
        
        episodes = []
        for episode_id, steps in steps_by_episode.items():
            steps.sort(key=lambda s: s.step)
            meta = episode_metadata.get(episode_id, {})
            
            total_tokens = meta.get("total_tokens")
            if total_tokens is None:
                step_tokens = [s.tokens for s in steps if s.tokens is not None]
                total_tokens = sum(step_tokens) if step_tokens else None
            
            episode = Episode(
                episode_id=episode_id,
                steps=steps,
                success=meta.get("success"),
                reward=meta.get("reward"),
                total_tokens=total_tokens,
                total_cost=meta.get("total_cost"),
                metadata=meta.get("metadata", {}),
            )
            episodes.append(episode)
        
        episodes.sort(key=lambda e: e.episode_id)
        return cls(episodes)
    
    def __len__(self) -> int:
        return len(self._episodes)
    
    def __iter__(self) -> Iterator[Episode]:
        return iter(self._episodes)
    
    def __getitem__(self, idx: Union[int, str]) -> Episode:
        if isinstance(idx, str):
            return self._by_id[idx]
        return self._episodes[idx]
    
    @property
    def episodes(self) -> List[Episode]:
        """Get all episodes."""
        return self._episodes
    
    def filter(self, predicate: Callable[[Episode], bool]) -> "AgentRunLog":
        """Filter episodes by predicate, return new AgentRunLog."""
        filtered = [e for e in self._episodes if predicate(e)]
        return AgentRunLog(filtered)
    
    def split(
        self, 
        predicate: Callable[[Episode], bool]
    ) -> tuple["AgentRunLog", "AgentRunLog"]:
        """Split into two AgentRunLogs based on predicate.
        
        Returns (matches, non_matches).
        """
        matches = []
        non_matches = []
        for e in self._episodes:
            if predicate(e):
                matches.append(e)
            else:
                non_matches.append(e)
        return AgentRunLog(matches), AgentRunLog(non_matches)
    
    def to_jsonl(self, path: Union[str, Path]) -> None:
        """Write traces to JSONL file."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for episode in self._episodes:
                for step in episode.steps:
                    f.write(json.dumps(step.to_dict()) + "\n")
                # Write episode_end if not already present
                has_end = any(s.type == StepType.EPISODE_END for s in episode.steps)
                if not has_end:
                    end_step = {
                        "episode_id": episode.episode_id,
                        "step": len(episode.steps),
                        "type": "episode_end",
                        "success": episode.success,
                        "reward": episode.reward,
                        "total_tokens": episode.total_tokens,
                        "total_cost": episode.total_cost,
                    }
                    f.write(json.dumps(end_step) + "\n")
    
    # Aggregate statistics
    @property
    def success_rate(self) -> float:
        """Fraction of episodes that succeeded."""
        with_status = [e for e in self._episodes if e.success is not None]
        if not with_status:
            return 0.0
        return sum(1 for e in with_status if e.success) / len(with_status)
    
    @property 
    def avg_steps(self) -> float:
        """Average number of steps per episode."""
        if not self._episodes:
            return 0.0
        return sum(e.total_steps for e in self._episodes) / len(self._episodes)
    
    @property
    def avg_tokens(self) -> Optional[float]:
        """Average tokens per episode (None if no token data)."""
        with_tokens = [e.total_tokens for e in self._episodes if e.total_tokens is not None]
        if not with_tokens:
            return None
        return sum(with_tokens) / len(with_tokens)
    
    @property
    def total_cost(self) -> Optional[float]:
        """Total cost across all episodes (None if no cost data)."""
        with_cost = [e.total_cost for e in self._episodes if e.total_cost is not None]
        if not with_cost:
            return None
        return sum(with_cost)
    
    @property
    def tool_error_rate(self) -> float:
        """Overall tool error rate across all episodes."""
        total_results = 0
        total_errors = 0
        for e in self._episodes:
            total_results += len(e.tool_results)
            total_errors += len(e.tool_errors)
        if total_results == 0:
            return 0.0
        return total_errors / total_results
