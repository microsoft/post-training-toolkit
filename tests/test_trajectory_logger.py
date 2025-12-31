from __future__ import annotations

from pathlib import Path

import pytest

from post_training_toolkit.agents import AgentRunLog
from post_training_toolkit.agents.trajectory import TrajectoryLogger


def test_logger_writes_episode_end_on_normal_exit(tmp_path: Path) -> None:
    out = tmp_path / "runs.jsonl"

    with TrajectoryLogger(out, auto_diagnostics=False) as logger:
        with logger.episode(episode_id="ep_1") as ep:
            ep.user("hi")
            ep.assistant("hello")
            ep.end(success=True, reward=1.0)

    runs = AgentRunLog.from_jsonl(out)
    assert len(runs) == 1
    assert runs["ep_1"].success is True
    assert any(s.type.value == "episode_end" for s in runs["ep_1"].steps)


def test_logger_auto_ends_episode_when_user_forgets(tmp_path: Path) -> None:
    out = tmp_path / "runs.jsonl"

    with TrajectoryLogger(out, auto_diagnostics=False) as logger:
        with logger.episode(episode_id="ep_1", success_on_exit=True) as ep:
            ep.user("task")
            ep.assistant("done")

    runs = AgentRunLog.from_jsonl(out)
    assert len(runs) == 1
    assert runs["ep_1"].success is True
    assert any(s.type.value == "episode_end" for s in runs["ep_1"].steps)


def test_logger_marks_failure_on_exception(tmp_path: Path) -> None:
    out = tmp_path / "runs.jsonl"

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with TrajectoryLogger(out, auto_diagnostics=False) as logger:
            with logger.episode(episode_id="ep_1") as ep:
                ep.user("task")
                raise Boom("nope")

    runs = AgentRunLog.from_jsonl(out)
    assert len(runs) == 1
    assert runs["ep_1"].success is False
    assert any(s.type.value == "episode_end" for s in runs["ep_1"].steps)


def test_call_tool_logs_tool_call_and_result(tmp_path: Path) -> None:
    out = tmp_path / "runs.jsonl"

    def my_tool(x: int) -> dict:
        return {"x": x, "ok": True}

    with TrajectoryLogger(out, auto_diagnostics=False) as logger:
        with logger.episode(episode_id="ep_1") as ep:
            ep.user("task")
            result = ep.call_tool(my_tool, tool_name="my_tool", args={"x": 3})
            assert result == {"x": 3, "ok": True}
            ep.end(success=True)

    runs = AgentRunLog.from_jsonl(out)
    ep = runs["ep_1"]
    tool_calls = [s for s in ep.steps if s.type.value == "tool_call"]
    tool_results = [s for s in ep.steps if s.type.value == "tool_result"]
    assert len(tool_calls) == 1
    assert len(tool_results) == 1
    assert tool_calls[0].tool == "my_tool"
    assert tool_results[0].tool == "my_tool"
    assert tool_results[0].error is None
    assert tool_results[0].result is not None


def test_call_tool_logs_tool_error(tmp_path: Path) -> None:
    out = tmp_path / "runs.jsonl"

    def bad_tool() -> None:
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError):
        with TrajectoryLogger(out, auto_diagnostics=False) as logger:
            with logger.episode(episode_id="ep_1") as ep:
                ep.user("task")
                ep.call_tool(bad_tool, tool_name="bad_tool")

    runs = AgentRunLog.from_jsonl(out)
    ep = runs["ep_1"]
    tool_results = [s for s in ep.steps if s.type.value == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].tool == "bad_tool"
    assert tool_results[0].error is not None
    assert any(s.type.value == "episode_end" for s in ep.steps)
