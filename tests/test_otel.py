"""§3.1: OpenTelemetry spans following the GenAI semantic conventions.

These assert on real exported span content via an in-memory exporter, not merely that
the instrumentation runs without raising.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import asyncpg
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agent_harness.agent import Agent
from agent_harness.checkpointer import Checkpointer
from agent_harness.executor import Executor
from agent_harness.models import Completion, ToolCall
from agent_harness.otel import (
    AGENT_COST_USD,
    AGENT_RUN_ID,
    AGENT_STEP,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    configure_tracing,
)
from agent_harness.tools import build_registry
from agent_harness.tracer import Tracer
from helpers import MockLLMClient


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    """Install a temporary in-memory tracer provider and hand back the exporter.

    Saves the *private* `_TRACER_PROVIDER` rather than `get_tracer_provider()`: the
    latter returns a proxy that delegates back to the global, so restoring it would
    make the global point at a proxy of itself and recurse forever on the next lookup.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = getattr(trace, "_TRACER_PROVIDER", None)
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    try:
        yield exporter
    finally:
        trace._TRACER_PROVIDER = previous  # type: ignore[attr-defined]


def _by_name(exporter: InMemorySpanExporter, prefix: str) -> list[object]:
    return [s for s in exporter.get_finished_spans() if s.name.startswith(prefix)]


async def test_llm_span_carries_genai_attributes(
    db_pool: asyncpg.Pool, spans: InMemorySpanExporter
) -> None:
    cp = Checkpointer(db_pool)
    tracer = Tracer(db_pool, model="claude-sonnet-4-5")
    state = await cp.create_run("t")
    comp = Completion(
        stop_reason="end_turn", text="hi", prompt_tokens=120, output_tokens=34
    )
    await tracer.llm(state, comp, latency_ms=42, cost_usd=0.00087)

    llm_spans = _by_name(spans, "chat ")
    assert len(llm_spans) == 1
    attrs = llm_spans[0].attributes  # type: ignore[attr-defined]
    assert llm_spans[0].name == "chat claude-sonnet-4-5"  # type: ignore[attr-defined]
    assert attrs[GEN_AI_SYSTEM] == "anthropic"
    assert attrs[GEN_AI_OPERATION_NAME] == "chat"
    assert attrs[GEN_AI_REQUEST_MODEL] == "claude-sonnet-4-5"
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 120
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 34
    assert tuple(attrs[GEN_AI_RESPONSE_FINISH_REASONS]) == ("end_turn",)
    assert attrs[AGENT_COST_USD] == pytest.approx(0.00087)
    assert attrs[AGENT_RUN_ID] == str(state.run_id)


async def test_tool_span_carries_tool_attributes(
    db_pool: asyncpg.Pool, spans: InMemorySpanExporter
) -> None:
    from agent_harness.models import ToolResult

    cp = Checkpointer(db_pool)
    tracer = Tracer(db_pool)
    state = await cp.create_run("t")
    await tracer.tool(
        state, ToolResult(call_id="tu_1", output="hello"), tool_name="read_file",
        latency_ms=7,
    )

    tool_spans = _by_name(spans, "execute_tool ")
    assert len(tool_spans) == 1
    attrs = tool_spans[0].attributes  # type: ignore[attr-defined]
    assert tool_spans[0].name == "execute_tool read_file"  # type: ignore[attr-defined]
    assert attrs[GEN_AI_OPERATION_NAME] == "execute_tool"
    assert attrs[GEN_AI_TOOL_NAME] == "read_file"
    assert attrs["agent.tool.output_bytes"] == 5
    assert tool_spans[0].status.status_code != StatusCode.ERROR  # type: ignore[attr-defined]


async def test_failed_tool_sets_span_error_status(
    db_pool: asyncpg.Pool, spans: InMemorySpanExporter
) -> None:
    """A tool error is data to the loop, but must still be visible as a span error."""
    from agent_harness.models import ToolResult

    cp = Checkpointer(db_pool)
    tracer = Tracer(db_pool)
    state = await cp.create_run("t")
    await tracer.tool(
        state,
        ToolResult(call_id="tu_1", error="read_file failed: FileNotFoundError: nope"),
        tool_name="read_file",
        latency_ms=3,
    )

    span = _by_name(spans, "execute_tool ")[0]
    assert span.status.status_code == StatusCode.ERROR  # type: ignore[attr-defined]
    assert "FileNotFoundError" in span.status.description  # type: ignore[attr-defined]
    assert span.attributes["error.type"] == "read_file failed"  # type: ignore[attr-defined]


async def test_full_run_emits_one_span_per_llm_and_tool_call(
    db_pool: asyncpg.Pool, spans: InMemorySpanExporter, temp_repo: Path
) -> None:
    """Span counts must match the run_events rows: same data, two destinations."""
    registry = build_registry()
    cp = Checkpointer(db_pool)
    tracer = Tracer(db_pool, model="test-model")
    agent = Agent(
        llm=MockLLMClient(
            [
                Completion(
                    stop_reason="tool_use",
                    prompt_tokens=100,
                    output_tokens=20,
                    tool_calls=[
                        ToolCall(
                            id="t1", name="read_file",
                            input={"path": str(temp_repo / "config.txt")},
                        ),
                        ToolCall(id="t2", name="list_dir", input={"path": str(temp_repo)}),
                    ],
                ),
                Completion(stop_reason="end_turn", text="done", prompt_tokens=200,
                           output_tokens=30),
            ]
        ),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
        tracer=tracer,
        max_steps=5,
    )
    state = await cp.create_run("read the config")
    await agent.run(state)

    assert len(_by_name(spans, "chat ")) == 2
    assert len(_by_name(spans, "execute_tool ")) == 2

    # Cross-check against the durable rows: the two views must agree.
    events = await tracer.timeline(state.run_id)
    assert sum(1 for e in events if e["kind"] == "llm_call") == 2
    assert sum(1 for e in events if e["kind"] == "tool_call") == 2


async def test_steps_are_labelled_on_spans(
    db_pool: asyncpg.Pool, spans: InMemorySpanExporter, temp_repo: Path
) -> None:
    """Step numbers on spans are what make a multi-step run readable in a trace UI."""
    registry = build_registry()
    cp = Checkpointer(db_pool)
    tracer = Tracer(db_pool, model="m")
    agent = Agent(
        llm=MockLLMClient(
            [
                Completion(
                    stop_reason="tool_use",
                    tool_calls=[ToolCall(id="t1", name="list_dir",
                                         input={"path": str(temp_repo)})],
                ),
                Completion(stop_reason="end_turn", text="done"),
            ]
        ),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        checkpointer=cp,
        tracer=tracer,
    )
    state = await cp.create_run("t")
    await agent.run(state)

    steps = [s.attributes[AGENT_STEP] for s in _by_name(spans, "chat ")]  # type: ignore[attr-defined]
    assert steps == [0, 1]


def test_configure_tracing_rejects_unknown_exporter() -> None:
    import agent_harness.otel as otel_mod

    saved = otel_mod._INITIALIZED
    otel_mod._INITIALIZED = False
    try:
        with pytest.raises(ValueError, match="unknown OTEL exporter"):
            configure_tracing(exporter="carrier-pigeon")
    finally:
        otel_mod._INITIALIZED = saved


def test_configure_tracing_none_is_a_cheap_noop() -> None:
    """Default config must not install an exporter: instrumentation costs nothing unused."""
    import agent_harness.otel as otel_mod

    saved = otel_mod._INITIALIZED
    otel_mod._INITIALIZED = False
    try:
        configure_tracing(exporter="none")
        assert otel_mod._INITIALIZED is True
    finally:
        otel_mod._INITIALIZED = saved
