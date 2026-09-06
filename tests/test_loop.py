"""M2/M3: the agent loop end to end against a scripted MockLLMClient."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from agent_harness.agent import Agent, MaxStepsExceeded
from agent_harness.executor import Executor
from agent_harness.models import Completion, State, ToolCall
from agent_harness.tools import build_registry
from helpers import MockLLMClient


def _state(task: str = "read the config") -> State:
    return State(run_id=uuid4(), task=task, messages=[{"role": "user", "content": task}])  # type: ignore[list-item]


def _agent(llm: MockLLMClient, max_steps: int = 5) -> Agent:
    registry = build_registry()
    return Agent(
        llm=llm,
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=1),
        max_steps=max_steps,
    )


async def test_agent_reads_file_then_answers(temp_repo: Path) -> None:
    path = str(temp_repo / "config.txt")
    llm = MockLLMClient(
        [
            Completion(
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t1", name="read_file", input={"path": path})],
            ),
            Completion(stop_reason="end_turn", text="the port is 8080"),
        ]
    )
    agent = _agent(llm)
    state = _state()
    answer = await agent.run(state)

    assert answer == "the port is 8080"
    assert state.status == "completed"
    assert state.step_count == 1
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert "port=8080" in tool_msg.content
    assert tool_msg.tool_call_id == "t1"
    # The tool output was actually fed back to the model on the second call.
    assert any("port=8080" in m.content for m in llm.calls[1])


async def test_unknown_tool_becomes_error_result_not_exception() -> None:
    llm = MockLLMClient(
        [
            Completion(
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t1", name="does_not_exist", input={})],
            ),
            Completion(stop_reason="end_turn", text="recovered"),
        ]
    )
    state = _state()
    answer = await _agent(llm).run(state)
    assert answer == "recovered"
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert tool_msg.content.startswith("ERROR: unknown tool")


async def test_invalid_tool_input_becomes_error_result() -> None:
    llm = MockLLMClient(
        [
            Completion(
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t1", name="read_file", input={"wrong": 1})],
            ),
            Completion(stop_reason="end_turn", text="recovered"),
        ]
    )
    state = _state()
    answer = await _agent(llm).run(state)
    assert answer == "recovered"
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert tool_msg.content.startswith("ERROR: invalid input for read_file")


async def test_failing_tool_does_not_crash_the_loop(tmp_path: Path) -> None:
    llm = MockLLMClient(
        [
            Completion(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t1", name="read_file", input={"path": str(tmp_path / "ghost")})
                ],
            ),
            Completion(stop_reason="end_turn", text="file is missing"),
        ]
    )
    state = _state()
    assert await _agent(llm).run(state) == "file is missing"
    assert next(m for m in state.messages if m.role == "tool").content.startswith("ERROR:")


async def test_max_steps_cap_raises_and_persists_failed_status() -> None:
    """M3: a model that never stops trips MaxStepsExceeded at the cap."""
    never_stops = Completion(
        stop_reason="tool_use",
        tool_calls=[ToolCall(id="t1", name="list_dir", input={"path": "."})],
    )
    llm = MockLLMClient([never_stops], repeat_last=True)
    agent = _agent(llm, max_steps=3)
    state = _state()
    with pytest.raises(MaxStepsExceeded):
        await agent.run(state)
    assert state.status == "failed"
    assert state.step_count == 3


async def test_multiple_tool_calls_in_one_step_are_all_answered(temp_repo: Path) -> None:
    llm = MockLLMClient(
        [
            Completion(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        id="a", name="read_file",
                        input={"path": str(temp_repo / "config.txt")},
                    ),
                    ToolCall(id="b", name="list_dir", input={"path": str(temp_repo)}),
                ],
            ),
            Completion(stop_reason="end_turn", text="done"),
        ]
    )
    state = _state()
    assert await _agent(llm).run(state) == "done"
    tool_msgs = [m for m in state.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["a", "b"]


async def test_timed_out_tool_lets_the_loop_continue() -> None:
    """M3: a tool that sleeps past the timeout yields an error result, loop continues."""
    import asyncio

    from pydantic import BaseModel

    from agent_harness.tools.base import Registry, Tool

    class SlowInput(BaseModel):
        pass

    class SlowTool(Tool):
        name = "slow"
        description = "too slow"
        InputModel = SlowInput

        async def execute(self, args: BaseModel) -> str:
            await asyncio.sleep(5)
            return "never"

    registry = Registry()
    registry.register(SlowTool())
    llm = MockLLMClient(
        [
            Completion(
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t1", name="slow", input={})],
            ),
            Completion(stop_reason="end_turn", text="moved on"),
        ]
    )
    agent = Agent(
        llm=llm,
        registry=registry,
        executor=Executor(registry, timeout=0.05, max_retries=0),
        max_steps=5,
    )
    state = _state()
    assert await agent.run(state) == "moved on"
    assert state.status == "completed"
    assert next(m for m in state.messages if m.role == "tool").content == "ERROR: slow timed out"
