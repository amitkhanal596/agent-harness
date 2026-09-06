"""OpenTelemetry tracing using the GenAI semantic conventions.

The Postgres `run_events` table stays the system of record for `agent trace` — it is
queryable, durable, and owned by this project. OTel spans are emitted *alongside* it so
the same data reaches Grafana / Datadog / Honeycomb / Langfuse without those tools
needing to understand a bespoke schema.

Attribute names follow the OpenTelemetry GenAI semantic conventions (`gen_ai.*`), which
is what makes the spans interoperable rather than merely present.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span, Status, StatusCode

# --- GenAI semantic convention attribute names ---------------------------------------
# Spelled out as constants rather than inlined so a convention bump is a one-line change.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"

# Cost is not yet in the GenAI conventions; namespace it so it is obviously ours.
AGENT_COST_USD = "agent.cost.usd"
AGENT_RUN_ID = "agent.run.id"
AGENT_STEP = "agent.step"

_INITIALIZED = False


def configure_tracing(
    service_name: str = "agent-harness",
    exporter: str | None = None,
    endpoint: str | None = None,
) -> None:
    """Install a global tracer provider. Idempotent; safe to call from every entrypoint.

    `exporter` selects the backend: "none" (default — spans are created but dropped, so
    instrumentation costs nothing when unused), "console" for local debugging, or "otlp"
    to ship to a collector at `endpoint`. Chosen by config so no code changes when a
    deployment gains a collector.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    choice = (exporter or os.environ.get("OTEL_EXPORTER", "none")).lower()
    if choice == "none":
        _INITIALIZED = True
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if choice == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif choice == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        target = endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=target)))
    else:
        raise ValueError(f"unknown OTEL exporter: {choice!r} (use none|console|otlp)")

    trace.set_tracer_provider(provider)
    _INITIALIZED = True


def get_tracer() -> trace.Tracer:
    """The tracer for this package."""
    return trace.get_tracer("agent_harness")


@contextmanager
def run_span(run_id: Any, task: str) -> Iterator[Span]:
    """Root span covering one agent run, so every step nests under a single trace."""
    with get_tracer().start_as_current_span("agent.run") as span:
        span.set_attribute(AGENT_RUN_ID, str(run_id))
        span.set_attribute(GEN_AI_OPERATION_NAME, "invoke_agent")
        span.set_attribute("agent.task", task[:512])
        yield span


@contextmanager
def llm_span(model: str, run_id: Any, step: int) -> Iterator[Span]:
    """Span for one model call, named per the GenAI convention (`chat {model}`)."""
    with get_tracer().start_as_current_span(f"chat {model}") as span:
        span.set_attribute(GEN_AI_SYSTEM, "anthropic")
        span.set_attribute(GEN_AI_OPERATION_NAME, "chat")
        span.set_attribute(GEN_AI_REQUEST_MODEL, model)
        span.set_attribute(AGENT_RUN_ID, str(run_id))
        span.set_attribute(AGENT_STEP, step)
        yield span


def record_llm_result(
    span: Span,
    stop_reason: str,
    prompt_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    """Attach token usage, finish reason, and cost to a model-call span."""
    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [stop_reason])
    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
    span.set_attribute(AGENT_COST_USD, cost_usd)


@contextmanager
def tool_span(tool_name: str, call_id: str, run_id: Any, step: int) -> Iterator[Span]:
    """Span for one tool invocation (`execute_tool {name}` per the convention)."""
    with get_tracer().start_as_current_span(f"execute_tool {tool_name}") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, "execute_tool")
        span.set_attribute(GEN_AI_TOOL_NAME, tool_name)
        span.set_attribute(GEN_AI_TOOL_CALL_ID, call_id)
        span.set_attribute(AGENT_RUN_ID, str(run_id))
        span.set_attribute(AGENT_STEP, step)
        yield span


def record_error(span: Span, error: str) -> None:
    """Mark a span as failed with `error` as the reason.

    Tool errors are data to the loop but still deserve error status on the span, because
    that is what makes a failing tool visible on a latency/error dashboard.
    """
    span.set_status(Status(StatusCode.ERROR, error))
    span.set_attribute("error.type", error.split(":", 1)[0][:128])
