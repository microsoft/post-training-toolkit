from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import re

class StepType(str, Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_PARSE_ERROR = "tool_parse_error"
    EPISODE_END = "episode_end"
    OTHER = "other"
    
    @classmethod
    def from_str(cls, s: str) -> "StepType":
        try:
            return cls(s)
        except ValueError:
            return cls.OTHER

@dataclass
class Step:
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
        return self.type == StepType.TOOL_RESULT and self.error is not None

@dataclass
class Episode:
    episode_id: str
    steps: List[Step] = field(default_factory=list)
    success: Optional[bool] = None
    reward: Optional[float] = None
    total_tokens: Optional[int] = None
    total_cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _iter_text_fields(self) -> Iterator[str]:
        for s in self.steps:
            if s.content:
                yield s.content
            if s.result:
                yield s.result
            if s.tool:
                yield s.tool
            if s.args:
                for v in s.args.values():
                    if isinstance(v, str) and v:
                        yield v

    @property
    def parse_error_steps(self) -> List[Step]:
        out: List[Step] = []
        for s in self.steps:
            if s.type == StepType.TOOL_PARSE_ERROR:
                out.append(s)
                continue
            if s.metadata.get("parse_error") is True:
                out.append(s)
        return out

    @property
    def has_parse_error(self) -> bool:
        return len(self.parse_error_steps) > 0
    
    @property
    def total_steps(self) -> int:
        return len(self.steps)
    
    @property
    def user_messages(self) -> List[Step]:
        return [s for s in self.steps if s.type == StepType.USER_MESSAGE]
    
    @property
    def assistant_messages(self) -> List[Step]:
        return [s for s in self.steps if s.type == StepType.ASSISTANT_MESSAGE]
    
    @property
    def tool_calls(self) -> List[Step]:
        return [s for s in self.steps if s.type == StepType.TOOL_CALL]
    
    @property
    def tool_results(self) -> List[Step]:
        return [s for s in self.steps if s.type == StepType.TOOL_RESULT]
    
    @property
    def tool_errors(self) -> List[Step]:
        return [s for s in self.steps if s.is_tool_error]
    
    @property
    def tool_error_rate(self) -> float:
        tool_results = self.tool_results
        if not tool_results:
            return 0.0
        return len(self.tool_errors) / len(tool_results)
    
    @property
    def initial_prompt(self) -> Optional[str]:
        user_msgs = self.user_messages
        return user_msgs[0].content if user_msgs else None
    
    @property
    def final_response(self) -> Optional[str]:
        asst_msgs = self.assistant_messages
        return asst_msgs[-1].content if asst_msgs else None
    
    def get_tool_call_sequence(self) -> List[str]:
        return [s.tool for s in self.tool_calls if s.tool]

    def _tool_call_arg_fingerprints(
        self,
        tool: Optional[str] = None,
        arg_key: Optional[str] = None,
    ) -> List[str]:
        fps: List[str] = []
        for s in self.tool_calls:
            if tool is not None and s.tool != tool:
                continue
            if not s.args:
                continue
            if arg_key is not None:
                val = s.args.get(arg_key)
                if val is None:
                    continue
                fps.append(str(val).strip())
            else:
                try:
                    fps.append(json.dumps(s.args, sort_keys=True, ensure_ascii=False))
                except TypeError:
                    fps.append(str(s.args))
        return fps

    def repeated_tool_call_args(
        self,
        tool: Optional[str] = None,
        arg_key: Optional[str] = None,
    ) -> List[str]:
        fps = self._tool_call_arg_fingerprints(tool=tool, arg_key=arg_key)
        seen: set[str] = set()
        repeated: set[str] = set()
        for fp in fps:
            if fp in seen:
                repeated.add(fp)
            else:
                seen.add(fp)
        return sorted(repeated)

    def repeated_query_fingerprints(
        self,
        tool: str = "search",
        arg_keys: Optional[List[str]] = None,
    ) -> List[str]:
        if arg_keys is None:
            arg_keys = ["q", "query"]
        repeated: set[str] = set()
        for key in arg_keys:
            repeated.update(self.repeated_tool_call_args(tool=tool, arg_key=key))
        return sorted(repeated)

    @property
    def max_consecutive_tool_calls(self) -> int:
        max_run = 0
        current = 0
        for s in self.steps:
            if s.type == StepType.TOOL_CALL:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

    def has_burst_tool_calls(self, max_consecutive_threshold: int = 5) -> bool:
        return self.max_consecutive_tool_calls > max_consecutive_threshold

    @staticmethod
    def _cjk_char_fraction(text: str) -> float:
        if not text:
            return 0.0
        cjk = re.findall(r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\uAC00-\uD7AF]", text)
        return len(cjk) / max(1, len(text))

    @property
    def cjk_char_rate(self) -> float:
        texts = list(self._iter_text_fields())
        if not texts:
            return 0.0
        joined = "\n".join(texts)
        return self._cjk_char_fraction(joined)
    
    def has_repeated_tool_pattern(self, min_repeats: int = 3) -> bool:
        tools = self.get_tool_call_sequence()
        if len(tools) < min_repeats:
            return False
        
        for i in range(len(tools) - min_repeats + 1):
            if len(set(tools[i:i + min_repeats])) == 1:
                return True
        
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
    
    def __init__(self, episodes: List[Episode]):
        self._episodes = episodes
        self._by_id = {e.episode_id: e for e in episodes}
    
    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> "AgentRunLog":
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
    
    @classmethod
    def from_episodes(cls, episodes: List[Episode]) -> "AgentRunLog":
        return cls(episodes)
    
    @classmethod
    def from_dicts(cls, records: List[Dict[str, Any]]) -> "AgentRunLog":
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
        return self._episodes
    
    def filter(self, predicate: Callable[[Episode], bool]) -> "AgentRunLog":
        filtered = [e for e in self._episodes if predicate(e)]
        return AgentRunLog(filtered)
    
    def split(
        self, 
        predicate: Callable[[Episode], bool]
    ) -> tuple["AgentRunLog", "AgentRunLog"]:
        matches = []
        non_matches = []
        for e in self._episodes:
            if predicate(e):
                matches.append(e)
            else:
                non_matches.append(e)
        return AgentRunLog(matches), AgentRunLog(non_matches)
    
    def to_jsonl(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for episode in self._episodes:
                for step in episode.steps:
                    f.write(json.dumps(step.to_dict()) + "\n")
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
    
    @property
    def success_rate(self) -> float:
        with_status = [e for e in self._episodes if e.success is not None]
        if not with_status:
            return 0.0
        return sum(1 for e in with_status if e.success) / len(with_status)
    
    @property 
    def avg_steps(self) -> float:
        if not self._episodes:
            return 0.0
        return sum(e.total_steps for e in self._episodes) / len(self._episodes)
    
    @property
    def avg_tokens(self) -> Optional[float]:
        with_tokens = [e.total_tokens for e in self._episodes if e.total_tokens is not None]
        if not with_tokens:
            return None
        return sum(with_tokens) / len(with_tokens)
    
    @property
    def total_cost(self) -> Optional[float]:
        with_cost = [e.total_cost for e in self._episodes if e.total_cost is not None]
        if not with_cost:
            return None
        return sum(with_cost)
    
    @property
    def tool_error_rate(self) -> float:
        total_results = 0
        total_errors = 0
        for e in self._episodes:
            total_results += len(e.tool_results)
            total_errors += len(e.tool_errors)
        if total_results == 0:
            return 0.0
        return total_errors / total_results
