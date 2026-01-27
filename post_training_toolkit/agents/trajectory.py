
from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()

def _json_safe(value: Any) -> Any:
	if value is None:
		return None
	if isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, (list, tuple)):
		return [_json_safe(v) for v in value]
	if isinstance(value, dict):
		return {str(k): _json_safe(v) for k, v in value.items()}
	return str(value)

def _stringify_result(value: Any) -> str:
	if value is None:
		return ""
	if isinstance(value, str):
		return value
	try:
		return json.dumps(value, ensure_ascii=False, default=str)
	except Exception:
		return str(value)

class TrajectoryLogger:
	def __init__(
		self,
		path: Union[str, Path],
		*,
		run_metadata: Optional[Dict[str, Any]] = None,
		flush_every: int = 1,
		auto_diagnostics: bool = True,
		diagnostics_budget_per_episode: Optional[float] = None,
		diagnostics_output: str = "stdout",
	) -> None:
		self.path = Path(path)
		self.path.parent.mkdir(parents=True, exist_ok=True)

		self.run_metadata = run_metadata or {}
		self.flush_every = max(1, int(flush_every))

		self.auto_diagnostics = auto_diagnostics
		self.diagnostics_budget_per_episode = diagnostics_budget_per_episode
		self.diagnostics_output = diagnostics_output

		self._fh = self.path.open("a", encoding="utf-8")
		self._write_count = 0
		self._closed = False

	def __enter__(self) -> "TrajectoryLogger":
		return self

	def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
		self.close()

	def episode(
		self,
		*,
		episode_id: Optional[str] = None,
		task: Optional[str] = None,
		success_on_exit: Optional[bool] = None,
		reward_on_exit: Optional[float] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> "EpisodeHandle":
		if episode_id is None:
			episode_id = f"ep_{uuid.uuid4().hex[:12]}"
		return EpisodeHandle(
			logger=self,
			episode_id=episode_id,
			task=task,
			success_on_exit=success_on_exit,
			reward_on_exit=reward_on_exit,
			episode_metadata=metadata or {},
		)

	def write_event(self, event: Dict[str, Any]) -> None:
		if self._closed:
			raise RuntimeError("TrajectoryLogger is closed")

		self._fh.write(json.dumps(_json_safe(event), ensure_ascii=False) + "\n")
		self._write_count += 1
		if self._write_count % self.flush_every == 0:
			self._fh.flush()

	def flush(self) -> None:
		if not self._closed:
			self._fh.flush()

	def close(self) -> None:
		if self._closed:
			return
		try:
			self._fh.flush()
		finally:
			self._fh.close()
			self._closed = True

		if self.auto_diagnostics:
			self._run_auto_diagnostics()

	def _run_auto_diagnostics(self) -> None:
		if self.diagnostics_output == "none":
			return

		try:
			from post_training_toolkit.agents.traces import AgentRunLog
			from post_training_toolkit.agents.heuristics import analyze_runs

			runs = AgentRunLog.from_jsonl(self.path)
			report = analyze_runs(runs, budget_per_episode=self.diagnostics_budget_per_episode)
			text = str(report)
		except Exception as e:
			text = f"[TrajectoryLogger] Auto-diagnostics failed: {e}"

		stream = sys.stdout if self.diagnostics_output == "stdout" else sys.stderr
		print(text, file=stream)

@dataclass
class EpisodeHandle:
	logger: TrajectoryLogger
	episode_id: str
	task: Optional[str] = None
	success_on_exit: Optional[bool] = None
	reward_on_exit: Optional[float] = None
	episode_metadata: Optional[Dict[str, Any]] = None

	def __post_init__(self) -> None:
		self._step = 0
		self._ended = False
		self._explicit_episode_end_metadata: Dict[str, Any] = {}

		if self.episode_metadata is None:
			self.episode_metadata = {}

		if self.task:
			self.user(self.task, metadata={"task": True})

	def __enter__(self) -> "EpisodeHandle":
		return self

	def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
		if self._ended:
			return

		success = self.success_on_exit
		reward = self.reward_on_exit

		if exc is not None:
			if success is None:
				success = False
			self.end(
				success=success,
				reward=reward,
				metadata={
					**self._explicit_episode_end_metadata,
					"exception_type": getattr(exc_type, "__name__", str(exc_type)),
					"exception": str(exc),
				},
			)
			return

		self.end(success=success, reward=reward, metadata=self._explicit_episode_end_metadata)

	def _emit(self, type_: str, **fields: Any) -> None:
		if self._ended:
			raise RuntimeError("Episode has already ended")

		event: Dict[str, Any] = {
			"episode_id": self.episode_id,
			"step": self._step,
			"type": type_,
			"timestamp": _utc_now_iso(),
		}
		self._step += 1

		for k, v in fields.items():
			if v is None:
				continue
			event[k] = v

		self.logger.write_event(event)

	def user(self, content: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
		if not isinstance(content, str) or not content:
			raise ValueError("user content must be a non-empty string")
		self._emit("user_message", content=content, metadata=metadata)

	def assistant(self, content: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
		if not isinstance(content, str) or not content:
			raise ValueError("assistant content must be a non-empty string")
		self._emit("assistant_message", content=content, metadata=metadata)

	def tool_call(
		self,
		tool: str,
		args: Optional[Dict[str, Any]] = None,
		*,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		if not isinstance(tool, str) or not tool:
			raise ValueError("tool must be a non-empty string")
		self._emit("tool_call", tool=tool, args=args, metadata=metadata)

	def tool_result(
		self,
		tool: str,
		*,
		result: Any = None,
		error: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		if not isinstance(tool, str) or not tool:
			raise ValueError("tool must be a non-empty string")

		result_str = None
		if result is not None:
			result_str = _stringify_result(result)

		self._emit("tool_result", tool=tool, result=result_str, error=error, metadata=metadata)

	def call_tool(
		self,
		tool_fn: Callable[..., Any],
		*,
		tool_name: Optional[str] = None,
		args: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None,
		reraise: bool = True,
	) -> Any:

		if tool_name is None:
			tool_name = getattr(tool_fn, "__name__", "tool")
		args = args or {}

		self.tool_call(tool_name, args=args, metadata=metadata)
		try:
			result = tool_fn(**args)
		except Exception as e:
			self.tool_result(tool_name, error=str(e), metadata=metadata)
			if reraise:
				raise
			return None

		self.tool_result(tool_name, result=result, metadata=metadata)
		return result

	def end(
		self,
		*,
		success: Optional[bool] = None,
		reward: Optional[float] = None,
		total_tokens: Optional[int] = None,
		total_cost: Optional[float] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		if self._ended:
			return

		if metadata:
			self._explicit_episode_end_metadata = dict(metadata)

		combined_metadata: Optional[Dict[str, Any]]
		if self.episode_metadata or metadata:
			combined_metadata = {
				**(self.episode_metadata or {}),
				**(metadata or {}),
			}
		else:
			combined_metadata = None

		self._emit(
			"episode_end",
			success=success,
			reward=reward,
			total_tokens=total_tokens,
			total_cost=total_cost,
			metadata=combined_metadata,
		)
		self._ended = True
