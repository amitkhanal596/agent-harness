"""A second provider implementation, proving `LLMClient` is genuinely provider-agnostic.

The `LLMClient` Protocol always had the right shape, but with only one implementation
the "swappable" claim was untested — and the internal model leaked Anthropic's vocabulary
(`tool_use` / `tool_result` blocks) into what was supposed to be a neutral interface.

`OpenAIClient` speaks the OpenAI chat-completions dialect, which differs from Anthropic's
in every way that matters:

* Tools are `{"type": "function", "function": {...}}`, not flat `{name, description,
  input_schema}`.
* OpenAI *does* have a `tool` role; Anthropic does not, and needs `tool_result` blocks
  inside a user message.
* Tool arguments arrive as a **JSON string** to parse, not a dict.
* Usage is `prompt_tokens` / `completion_tokens`, not `input_tokens` / `output_tokens`.
* Finish reasons are `tool_calls` / `stop`, not `tool_use` / `end_turn`.

Getting a second dialect working end to end is what turns "there is a Protocol" into
"the abstraction actually holds".
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .models import Completion, Message, ToolCall, ToolSpec

#: USD per 1M tokens. VERIFY against current pricing before relying on these.
OPENAI_PRICE_PER_1M_INPUT = 2.50
OPENAI_PRICE_PER_1M_OUTPUT = 10.00

#: Map OpenAI finish reasons onto the internal vocabulary so the loop stays uniform.
_STOP_REASONS = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Map internal messages to the OpenAI chat format.

    Unlike Anthropic, OpenAI has a real `tool` role, so results stay separate messages
    rather than being grouped into a synthetic user turn — the mirror image of the
    grouping logic in `llm.to_anthropic_messages`.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "user":
            out.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.input),
                        },
                    }
                    for call in msg.tool_calls
                ]
            out.append(entry)
        elif msg.role == "tool":
            out.append(
                {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
            )
        else:
            raise ValueError(f"unknown message role: {msg.role!r}")
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Wrap tool specs in OpenAI's nested `function` envelope."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def openai_cost_usd(prompt_tokens: int, output_tokens: int) -> float:
    """Cost of one OpenAI completion."""
    return (
        prompt_tokens / 1e6 * OPENAI_PRICE_PER_1M_INPUT
        + output_tokens / 1e6 * OPENAI_PRICE_PER_1M_OUTPUT
    )


class OpenAIClient:
    """`LLMClient` speaking the OpenAI chat-completions API.

    Works against OpenAI itself and any compatible server (llama.cpp, vLLM, Ollama,
    Together) by pointing `base_url` at it. HTTP errors propagate for the guardrails to
    classify.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = 4096,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> Completion:
        """Call the model once and normalize the response into a Completion."""
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": to_openai_messages(messages),
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()

        choice = body["choices"][0]
        message = choice["message"]
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw["function"]
            # Arguments arrive as a JSON *string*; a malformed one becomes an empty dict
            # so schema validation reports it as a tool error rather than crashing here.
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(ToolCall(id=raw["id"], name=function["name"], input=arguments))

        usage = body.get("usage") or {}
        return Completion(
            text=message.get("content") or "",
            tool_calls=calls,
            stop_reason=_STOP_REASONS.get(
                choice.get("finish_reason", ""), choice.get("finish_reason", "")
            ),
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
