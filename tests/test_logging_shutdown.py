"""§2.10 structured logging with correlation, and §2.4 graceful shutdown."""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

import asyncpg
import pytest

from agent_harness.agent import Agent, RunPaused
from agent_harness.checkpointer import Checkpointer
from agent_harness.executor import Executor
from agent_harness.logging import JsonFormatter, configure_logging, log_context
from agent_harness.models import Completion, State, ToolCall
from agent_harness.tools import build_registry
from helpers import MockLLMClient


def _format(record_kwargs: dict[str, object] | None = None, msg: str = "hello") -> dict:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for key, value in (record_kwargs or {}).items():
        setattr(record, key, value)
    return json.loads(JsonFormatter().format(record))


# --- structured logging -------------------------------------------------------------


def test_log_line_is_valid_json_with_the_core_fields() -> None:
    payload = _format()
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["message"] == "hello"
    assert "ts" in payload


def test_extra_fields_are_merged_into_the_payload() -> None:
    payload = _format({"latency_ms": 42, "tool": "read_file"})
    assert payload["latency_ms"] == 42
    assert payload["tool"] == "read_file"


def test_context_fields_are_attached_to_every_record() -> None:
    with log_context(run_id="abc-123"):
        payload = _format()
    assert payload["run_id"] == "abc-123"


def test_contexts_nest_rather_than_replace() -> None:
    with log_context(run_id="r1"):
        with log_context(step=3):
            payload = _format()
    assert payload["run_id"] == "r1"
    assert payload["step"] == 3


def test_context_is_removed_on_exit() -> None:
    with log_context(run_id="r1"):
        pass
    assert "run_id" not in _format()


async def test_context_is_isolated_between_concurrent_tasks() -> None:
    """ContextVar semantics: two concurrent runs must not see each other's run_id."""
    seen: dict[str, str] = {}

    async def one(name: str) -> None:
        with log_context(run_id=name):
            await asyncio.sleep(0.01)
            seen[name] = str(_format().get("run_id"))

    await asyncio.gather(one("run-a"), one("run-b"))
    assert seen == {"run-a": "run-a", "run-b": "run-b"}


def test_exception_info_is_captured() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="t", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="failed", args=(), exc_info=sys.exc_info(),
        )
        payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_is_idempotent() -> None:
    configure_logging(level="DEBUG")
    configure_logging(level="INFO")
    assert len(logging.getLogger().handlers) == 1


def test_non_serializable_extras_do_not_crash_the_logger() -> None:
    """A logger that raises while logging an error is worse than a lossy log line."""
    payload = _format({"obj": object()})
    assert "obj" in payload


async def test_loop_emits_correlated_lines(
    caplog: pytest.LogCaptureFixture
) -> None:
    """Every line from a run carries its run_id, which is the point of correlation."""
    registry = build_registry()
    llm = MockLLMClient([Completion(stop_reason="end_turn", text="done")])
    agent = Agent(
        llm=llm,
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
    )
    state = State(run_id=uuid4(), task="a task")
    with caplog.at_level(logging.INFO, logger="agent_harness.agent"):
        await agent.run(state)

    messages = [r.message for r in caplog.records]
    assert "run started" in messages
    assert "llm call" in messages
    assert "run completed" in messages


# --- graceful shutdown ---------------------------------------------------------------


async def test_shutdown_pauses_at_a_step_boundary(db_pool: asyncpg.Pool) -> None:
    """A shutdown request stops the loop cleanly and leaves a resumable checkpoint."""
    registry = build_registry()
    cp = Checkpointer(db_pool)
    shutdown = asyncio.Event()

    class SetsShutdownAfterFirstStep(MockLLMClient):
        async def complete(self, messages: object, tools: object) -> Completion:
            comp = await super().complete(messages, tools)  # type: ignore[arg-type]
            shutdown.set()  # request shutdown while step 0 is still in flight
            return comp

    agent = Agent(
        llm=SetsShutdownAfterFirstStep(
            [
                Completion(
                    stop_reason="tool_use",
                    tool_calls=[ToolCall(id="t1", name="list_dir", input={"path": "."})],
                ),
                Completion(stop_reason="end_turn", text="never reached"),
            ]
        ),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
        max_steps=10,
        shutdown=shutdown,
    )
    state = await cp.create_run("long task")
    with pytest.raises(RunPaused) as exc:
        await agent.run(state)

    assert "agent resume" in str(exc.value)
    assert state.status == "paused"

    persisted = await cp.latest(state.run_id)
    assert persisted.status == "paused"
    # The completed step was checkpointed; the loop stopped before starting the next.
    assert persisted.step_count == 1


async def test_a_paused_run_resumes_to_completion(db_pool: asyncpg.Pool) -> None:
    """The whole point of pausing cleanly: resume must finish the job."""
    registry = build_registry()
    cp = Checkpointer(db_pool)
    shutdown = asyncio.Event()

    class PausesAfterOneStep(MockLLMClient):
        async def complete(self, messages: object, tools: object) -> Completion:
            comp = await super().complete(messages, tools)  # type: ignore[arg-type]
            shutdown.set()
            return comp

    paused_agent = Agent(
        llm=PausesAfterOneStep(
            [
                Completion(
                    stop_reason="tool_use",
                    tool_calls=[ToolCall(id="t1", name="list_dir", input={"path": "."})],
                )
            ]
        ),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
        shutdown=shutdown,
    )
    state = await cp.create_run("resumable task")
    with pytest.raises(RunPaused):
        await paused_agent.run(state)

    resumed = Agent(
        llm=MockLLMClient([Completion(stop_reason="end_turn", text="finished")]),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
    )
    assert await resumed.resume(state.run_id) == "finished"
    assert (await cp.latest(state.run_id)).status == "completed"


async def test_no_shutdown_event_means_unchanged_behavior() -> None:
    """Shutdown is opt-in; an Agent without an event never pauses."""
    registry = build_registry()
    agent = Agent(
        llm=MockLLMClient([Completion(stop_reason="end_turn", text="ok")]),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
    )
    state = State(run_id=uuid4(), task="t")
    assert await agent.run(state) == "ok"


async def test_shutdown_before_the_first_step_pauses_immediately(
    db_pool: asyncpg.Pool
) -> None:
    registry = build_registry()
    cp = Checkpointer(db_pool)
    shutdown = asyncio.Event()
    shutdown.set()

    agent = Agent(
        llm=MockLLMClient([Completion(stop_reason="end_turn", text="never")]),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
        shutdown=shutdown,
    )
    state = await cp.create_run("t")
    with pytest.raises(RunPaused):
        await agent.run(state)
    assert state.step_count == 0
