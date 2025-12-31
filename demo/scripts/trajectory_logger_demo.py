"""Demo: TrajectoryLogger tiny generic fallback + auto-diagnostics.

Runs a toy "agent loop" for several episodes, writes trace v1 JSONL, and
prints the agent diagnostics report automatically when the logger closes.

Run:
    python demo/scripts/trajectory_logger_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from post_training_toolkit import TrajectoryLogger


def search(query: str) -> str:
    if "error" in query:
        raise RuntimeError("search backend unavailable")
    return f"results_for={query}"  # toy


def book(item_id: str) -> str:
    if item_id == "bad":
        raise RuntimeError("booking failed")
    return f"confirmation={item_id}-XYZ"  # toy


def toy_agent_loop(task: str, ep) -> None:
    """A minimal custom loop: assistant decides tools, tools return results."""

    ep.user(task)

    # "Policy": choose behavior based on task string
    if "loop" in task:
        ep.assistant("I will keep trying the same tool.")
        for _ in range(3):
            # repeated tool calls trigger loop heuristics
            ep.call_tool(search, tool_name="search", args={"query": "same"}, reraise=False)
        ep.assistant("Still stuck.")
        ep.end(success=False, reward=0.0)
        return

    if "tool_error" in task:
        ep.assistant("I'll search, but it might fail.")
        ep.call_tool(search, tool_name="search", args={"query": "error please"}, reraise=False)
        ep.assistant("I couldn't complete it.")
        ep.end(success=False, reward=0.0)
        return

    if "book_bad" in task:
        ep.assistant("Attempting to book.")
        ep.call_tool(book, tool_name="book", args={"item_id": "bad"}, reraise=False)
        ep.end(success=False, reward=0.0)
        return

    # success path
    ep.assistant("Searching and booking now.")
    result = ep.call_tool(search, tool_name="search", args={"query": "paris"}, reraise=True)
    ep.assistant(f"Found: {result}. Booking.")
    confirmation = ep.call_tool(book, tool_name="book", args={"item_id": "AA123"}, reraise=True)
    ep.assistant(f"Done: {confirmation}")
    ep.end(success=True, reward=1.0)


def main() -> None:
    out_dir = Path(__file__).parent.parent / "outputs" / "trajectory_logger_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "runs.jsonl"

    print("=" * 70)
    print("TRAJECTORY LOGGER DEMO")
    print("=" * 70)
    print(f"Writing traces to: {traces_path}")
    print("(Diagnostics report will print automatically at the end.)")
    print()

    tasks = [
        "book flight to paris",  # success
        "book flight to paris again",  # success
        "tool_error: flaky search",  # tool error
        "book_bad: tool error",  # tool error
        "loop: agent stuck",  # loop
        "loop: agent stuck again",  # loop
    ]

    # Auto-diagnostics is ON by default; report prints when logger closes.
    with TrajectoryLogger(traces_path, auto_diagnostics=True, diagnostics_output="stdout") as logger:
        for i, task in enumerate(tasks):
            with logger.episode(episode_id=f"ep_{i:03d}") as ep:
                toy_agent_loop(task, ep)


if __name__ == "__main__":
    main()
