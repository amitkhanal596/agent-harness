"""LLM client protocol, the Anthropic implementation, and cost accounting."""

from __future__ import annotations

from typing import Any, Protocol, cast

from anthropic import AsyncAnthropic

from .models import Completion, Message, ToolCall, ToolSpec

# Pricing in USD per 1M tokens. VERIFY against https://claude.com/pricing before relying
# on these for billing; they are documented constants, not a live feed.
PRICE_PER_1M_INPUT = 3.00
PRICE_PER_1M_OUTPUT = 15.00


def cost_usd(prompt_tokens: int, output_tokens: int) -> float:
    """Cost of one completion: tokens / 1e6 * price."""
    return (prompt_tokens / 1e6) * PRICE_PER_1M_INPUT + (output_tokens / 1e6) * PRICE_PER_1M_OUTPUT


class LLMClient(Protocol):
    """Anything that can turn conversation state plus tools into a Completion."""

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
        ...


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Map internal messages to the Anthropic messages format.

    Anthropic has no "tool" role: tool results are `tool_result` blocks inside a *user*
    message, and consecutive results are grouped into a single user message preserving
    order.
    """
    out: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for msg in messages:
        if msg.role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )
            continue

        flush()
        if msg.role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": msg.content}]})
        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
                )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        else:  # pragma: no cover - guards a programming error
            raise ValueError(f"unknown message role: {msg.role!r}")

    flush()
    return out


class AnthropicClient:
    """LLMClient backed by the Anthropic SDK.

    Network and API errors propagate to the caller; the guardrails layer decides whether
    to retry or trip the circuit breaker.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = 4096,
        base_url: str | None = None,
        cache_prompt: bool = False,
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url or None)
        self._model = model
        self._max_tokens = max_tokens
        self._cache_prompt = cache_prompt

    def _tool_payload(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Tool definitions, with a cache breakpoint on the last one when caching is on.

        Tool definitions are the longest stable prefix of every request in a run, so one
        breakpoint at their end lets the provider reuse them across iterations.
        """
        payload = [t.model_dump() for t in tools]
        if self._cache_prompt and payload:
            payload[-1]["cache_control"] = {"type": "ephemeral"}
        return payload

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
        """Call the model once and normalize the response into a Completion."""
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            tools=cast(Any, self._tool_payload(tools)),
            messages=cast(Any, to_anthropic_messages(messages)),
        )
        text = ""
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        return Completion(
            text=text,
            tool_calls=calls,
            stop_reason=resp.stop_reason or "",
            prompt_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
