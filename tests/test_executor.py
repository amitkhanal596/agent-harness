"""M3: executor timeouts, retry with backoff, and transient-vs-fatal classification."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from agent_harness.executor import Executor, FatalToolError, is_transient
from agent_harness.models import ToolCall
from agent_harness.tools.base import Registry, Tool


class SleepInput(BaseModel):
    seconds: float = 0.0


class SleepTool(Tool):
    name = "sleep_tool"
    description = "sleeps"
    InputModel = SleepInput

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, SleepInput)
        await asyncio.sleep(args.seconds)
        return f"slept {args.seconds}"


class FlakyInput(BaseModel):
    pass


class FlakyTool(Tool):
    """Fails `fail_times` times with a transient error, then succeeds."""

    name = "flaky"
    description = "fails then succeeds"
    InputModel = FlakyInput

    def __init__(self, fail_times: int, exc: Exception | None = None) -> None:
        self.fail_times = fail_times
        self.attempts = 0
        self._exc = exc or ConnectionError("temporary network blip")

    async def execute(self, args: BaseModel) -> str:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise self._exc
        return "succeeded"


class FatalTool(Tool):
    name = "fatal"
    description = "always fatal"
    InputModel = FlakyInput

    async def execute(self, args: BaseModel) -> str:
        raise FatalToolError("unrecoverable")


class StrictInput(BaseModel):
    required_field: int


class StrictTool(Tool):
    name = "strict"
    description = "needs a valid int"
    InputModel = StrictInput

    async def execute(self, args: BaseModel) -> str:
        return "ok"


def _executor(tools: list[Tool], timeout: float = 0.2, max_retries: int = 2) -> Executor:
    registry = Registry()
    for tool in tools:
        registry.register(tool)
    return Executor(registry, timeout=timeout, max_retries=max_retries)


async def test_unknown_tool_returns_error() -> None:
    result = await _executor([]).run(ToolCall(id="c1", name="ghost", input={}))
    assert result.error is not None
    assert "unknown tool" in result.error


async def test_invalid_input_returns_validation_error() -> None:
    result = await _executor([StrictTool()]).run(
        ToolCall(id="c1", name="strict", input={"required_field": "not-an-int"})
    )
    assert result.error is not None
    assert "invalid input for strict" in result.error


async def test_timeout_becomes_error_result() -> None:
    ex = _executor([SleepTool()], timeout=0.05, max_retries=0)
    result = await ex.run(ToolCall(id="c1", name="sleep_tool", input={"seconds": 1.0}))
    assert result.error == "sleep_tool timed out"
    assert result.output == ""


async def test_timeout_is_retried_up_to_max_retries() -> None:
    tool = SleepTool()
    ex = _executor([tool], timeout=0.02, max_retries=2)
    result = await ex.run(ToolCall(id="c1", name="sleep_tool", input={"seconds": 1.0}))
    assert result.error is not None
    assert "timed out" in result.error


async def test_fails_twice_then_succeeds() -> None:
    tool = FlakyTool(fail_times=2)
    result = await _executor([tool], max_retries=2).run(ToolCall(id="c1", name="flaky", input={}))
    assert result.error is None
    assert result.output == "succeeded"
    assert tool.attempts == 3


async def test_retries_are_bounded() -> None:
    tool = FlakyTool(fail_times=99)
    result = await _executor([tool], max_retries=2).run(ToolCall(id="c1", name="flaky", input={}))
    assert result.error is not None
    assert tool.attempts == 3  # initial + 2 retries


async def test_non_transient_error_is_not_retried() -> None:
    tool = FlakyTool(fail_times=99, exc=ValueError("bad data"))
    result = await _executor([tool], max_retries=2).run(ToolCall(id="c1", name="flaky", input={}))
    assert result.error is not None
    assert "ValueError" in result.error
    assert tool.attempts == 1


async def test_classification_helper() -> None:
    assert is_transient(TimeoutError())
    assert is_transient(ConnectionError())
    assert is_transient(OSError("io"))
    assert not is_transient(ValueError("nope"))
    assert not is_transient(FatalToolError("stop"))


async def test_fatal_tool_error_propagates_from_run() -> None:
    with pytest.raises(FatalToolError):
        await _executor([FatalTool()]).run(ToolCall(id="c1", name="fatal", input={}))


async def test_backoff_delay_grows(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_harness.executor as ex_mod

    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(d: float) -> None:
        delays.append(d)
        await real_sleep(0)

    monkeypatch.setattr(ex_mod.asyncio, "sleep", fake_sleep)
    tool = FlakyTool(fail_times=99)
    await _executor([tool], max_retries=3).run(ToolCall(id="c1", name="flaky", input={}))
    assert delays == [0.1, 0.2, 0.4]
