"""M6: run_all is concurrent, order-preserving, and cancels siblings on a fatal error."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import BaseModel

from agent_harness.agent import Agent
from agent_harness.executor import Executor, FatalToolError
from agent_harness.models import Completion, State, ToolCall
from agent_harness.tools.base import Registry, Tool
from helpers import MockLLMClient

SLEEP = 0.3


class SleepInput(BaseModel):
    seconds: float = SLEEP
    label: str = ""


class SleepTool(Tool):
    name = "sleeper"
    description = "sleeps then echoes its label"
    InputModel = SleepInput

    def __init__(self) -> None:
        self.cancelled = 0
        self.completed = 0

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, SleepInput)
        try:
            await asyncio.sleep(args.seconds)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.completed += 1
        return f"done:{args.label}"


class FatalInput(BaseModel):
    delay: float = 0.0


class FatalTool(Tool):
    name = "boom"
    description = "raises a fatal error"
    InputModel = FatalInput

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, FatalInput)
        await asyncio.sleep(args.delay)
        raise FatalToolError("unrecoverable failure")


def _executor(tools: list[Tool], timeout: float = 5.0, max_retries: int = 0) -> Executor:
    registry = Registry()
    for tool in tools:
        registry.register(tool)
    return Executor(registry, timeout=timeout, max_retries=max_retries)


async def test_three_sleeping_tools_run_in_about_one_sleep_not_three() -> None:
    ex = _executor([SleepTool()])
    calls = [
        ToolCall(id=f"c{i}", name="sleeper", input={"seconds": SLEEP, "label": str(i)})
        for i in range(3)
    ]
    t0 = time.perf_counter()
    results = await ex.run_all(calls)
    elapsed = time.perf_counter() - t0

    assert all(r.error is None for r in results)
    assert elapsed < SLEEP * 2, f"expected ~{SLEEP}s concurrent, took {elapsed:.2f}s"
    assert elapsed >= SLEEP * 0.8


async def test_results_preserve_input_order_regardless_of_finish_order() -> None:
    ex = _executor([SleepTool()])
    calls = [
        ToolCall(id="slow", name="sleeper", input={"seconds": 0.25, "label": "slow"}),
        ToolCall(id="fast", name="sleeper", input={"seconds": 0.01, "label": "fast"}),
        ToolCall(id="mid", name="sleeper", input={"seconds": 0.12, "label": "mid"}),
    ]
    results = await ex.run_all(calls)
    assert [r.call_id for r in results] == ["slow", "fast", "mid"]
    assert [r.output for r in results] == ["done:slow", "done:fast", "done:mid"]


async def test_fatal_error_cancels_siblings_and_keeps_alignment() -> None:
    sleeper = SleepTool()
    ex = _executor([sleeper, FatalTool()])
    calls = [
        ToolCall(id="a", name="sleeper", input={"seconds": 2.0, "label": "a"}),
        ToolCall(id="b", name="boom", input={"delay": 0.05}),
        ToolCall(id="c", name="sleeper", input={"seconds": 2.0, "label": "c"}),
    ]
    t0 = time.perf_counter()
    results = await ex.run_all(calls)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, "siblings should have been cancelled, not awaited"
    assert [r.call_id for r in results] == ["a", "b", "c"]
    assert results[0].error is not None and "cancelled" in results[0].error
    assert results[2].error is not None and "cancelled" in results[2].error
    assert sleeper.cancelled == 2
    assert sleeper.completed == 0


async def test_partial_results_are_retained_when_a_sibling_goes_fatal() -> None:
    sleeper = SleepTool()
    ex = _executor([sleeper, FatalTool()])
    calls = [
        ToolCall(id="quick", name="sleeper", input={"seconds": 0.01, "label": "quick"}),
        ToolCall(id="boom", name="boom", input={"delay": 0.2}),
        ToolCall(id="slow", name="sleeper", input={"seconds": 5.0, "label": "slow"}),
    ]
    results = await ex.run_all(calls)
    assert results[0].error is None and results[0].output == "done:quick"
    assert results[2].error is not None and "cancelled" in results[2].error


async def test_empty_call_list_returns_empty() -> None:
    assert await _executor([SleepTool()]).run_all([]) == []


async def test_mixed_success_and_error_all_returned() -> None:
    ex = _executor([SleepTool()])
    calls = [
        ToolCall(id="ok", name="sleeper", input={"seconds": 0.01, "label": "ok"}),
        ToolCall(id="ghost", name="not_registered", input={}),
        ToolCall(id="bad", name="sleeper", input={"seconds": "not-a-number"}),
    ]
    results = await ex.run_all(calls)
    assert [r.call_id for r in results] == ["ok", "ghost", "bad"]
    assert results[0].error is None
    assert results[1].error is not None and "unknown tool" in results[1].error
    assert results[2].error is not None and "invalid input" in results[2].error


async def test_loop_step_with_many_tools_is_concurrent() -> None:
    """The agent's multi-tool step benefits from the fan-out, not just the executor."""
    registry = Registry()
    registry.register(SleepTool())
    calls = [
        ToolCall(id=f"c{i}", name="sleeper", input={"seconds": SLEEP, "label": str(i)})
        for i in range(4)
    ]
    llm = MockLLMClient(
        [
            Completion(stop_reason="tool_use", tool_calls=calls),
            Completion(stop_reason="end_turn", text="all done"),
        ]
    )
    agent = Agent(llm=llm, registry=registry, executor=_executor([SleepTool()]), max_steps=5)
    from uuid import uuid4

    state = State(run_id=uuid4(), task="fan out")
    t0 = time.perf_counter()
    assert await agent.run(state) == "all done"
    elapsed = time.perf_counter() - t0
    assert elapsed < SLEEP * 2.5, f"4 x {SLEEP}s tools took {elapsed:.2f}s"
    assert len([m for m in state.messages if m.role == "tool"]) == 4


@pytest.mark.parametrize("count", [1, 5])
async def test_scales_to_n_tools(count: int) -> None:
    ex = _executor([SleepTool()])
    calls = [
        ToolCall(id=f"c{i}", name="sleeper", input={"seconds": 0.1, "label": str(i)})
        for i in range(count)
    ]
    t0 = time.perf_counter()
    results = await ex.run_all(calls)
    elapsed = time.perf_counter() - t0
    assert len(results) == count
    assert elapsed < 0.1 * count if count > 1 else True
