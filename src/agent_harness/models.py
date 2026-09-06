"""Pydantic data models shared across the harness. These are the wire/state contracts."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A model-requested tool invocation."""

    id: str
    name: str
    input: dict[str, Any]


class Message(BaseModel):
    """One turn of conversation state.

    `role` is "user" | "assistant" | "tool". Assistant turns carry `tool_calls` when the
    model requested tools; tool turns carry the `tool_call_id` they answer.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class State(BaseModel):
    """The complete, serializable state of a run. Checkpointed after every iteration."""

    run_id: UUID
    task: str
    messages: list[Message] = Field(default_factory=list)
    step_count: int = 0
    status: str = "running"


class ToolSpec(BaseModel):
    """A tool as advertised to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


class Completion(BaseModel):
    """One model response, normalized across providers."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str
    prompt_tokens: int = 0
    output_tokens: int = 0


class ToolResult(BaseModel):
    """The outcome of one tool call. A failure is data: `error` is set, never raised."""

    call_id: str
    output: str = ""
    error: str | None = None
