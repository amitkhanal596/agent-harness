"""M1: Completion parsing from a stubbed SDK response, and the message mapping."""

from __future__ import annotations

from typing import Any

import pytest

from agent_harness.llm import AnthropicClient, cost_usd, to_anthropic_messages
from agent_harness.models import Message, ToolCall


class _Block:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Response:
    def __init__(self, content: list[Any], stop_reason: str, usage: _Usage) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


class _StubMessages:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _Response:
        self.kwargs = kwargs
        return self._response


def _client_with(response: _Response) -> tuple[AnthropicClient, _StubMessages]:
    client = AnthropicClient(model="test-model", api_key="sk-test")
    stub = _StubMessages(response)
    client._client.messages = stub  # type: ignore[assignment]
    return client, stub


async def test_tool_use_block_maps_to_tool_calls() -> None:
    resp = _Response(
        content=[
            _Block(type="text", text="let me look"),
            _Block(type="tool_use", id="tu_1", name="read_file", input={"path": "a.txt"}),
        ],
        stop_reason="tool_use",
        usage=_Usage(120, 34),
    )
    client, _ = _client_with(resp)
    comp = await client.complete([Message(role="user", content="hi")], [])
    assert comp.text == "let me look"
    assert comp.tool_calls == [ToolCall(id="tu_1", name="read_file", input={"path": "a.txt"})]
    assert comp.stop_reason == "tool_use"
    assert (comp.prompt_tokens, comp.output_tokens) == (120, 34)


async def test_end_turn_text_block_maps_to_completion_text() -> None:
    resp = _Response(
        content=[_Block(type="text", text="the port is "), _Block(type="text", text="8080")],
        stop_reason="end_turn",
        usage=_Usage(10, 5),
    )
    client, _ = _client_with(resp)
    comp = await client.complete([Message(role="user", content="hi")], [])
    assert comp.text == "the port is 8080"
    assert comp.tool_calls == []
    assert comp.stop_reason == "end_turn"
    assert (comp.prompt_tokens, comp.output_tokens) == (10, 5)


def test_tool_role_serializes_to_user_tool_result_block() -> None:
    messages = [
        Message(role="user", content="read a.txt"),
        Message(
            role="assistant",
            content="ok",
            tool_calls=[ToolCall(id="tu_1", name="read_file", input={"path": "a.txt"})],
        ),
        Message(role="tool", tool_call_id="tu_1", content="hello"),
    ]
    out = to_anthropic_messages(messages)
    assert out[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "a.txt"}},
        ],
    }
    assert out[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "hello"}],
    }
    assert all(m["role"] != "tool" for m in out)


def test_consecutive_tool_results_are_grouped_in_one_user_message() -> None:
    messages = [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="a", name="t", input={}),
                ToolCall(id="b", name="t", input={}),
            ],
        ),
        Message(role="tool", tool_call_id="a", content="A"),
        Message(role="tool", tool_call_id="b", content="B"),
        Message(role="assistant", content="done"),
    ]
    out = to_anthropic_messages(messages)
    assert len(out) == 4
    assert out[2]["role"] == "user"
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["a", "b"]
    assert out[3] == {"role": "assistant", "content": [{"type": "text", "text": "done"}]}


def test_assistant_without_text_emits_only_tool_use_blocks() -> None:
    out = to_anthropic_messages(
        [Message(role="assistant", tool_calls=[ToolCall(id="x", name="t", input={"k": 1})])]
    )
    assert out == [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "t",
                                          "input": {"k": 1}}]}
    ]


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValueError):
        to_anthropic_messages([Message(role="system", content="nope")])


async def test_tools_and_max_tokens_are_sent() -> None:
    from agent_harness.models import ToolSpec

    resp = _Response([_Block(type="text", text="hi")], "end_turn", _Usage(1, 1))
    client, stub = _client_with(resp)
    spec = ToolSpec(name="t", description="d", input_schema={"type": "object"})
    await client.complete([Message(role="user", content="hi")], [spec])
    assert stub.kwargs["max_tokens"] == 4096
    assert stub.kwargs["model"] == "test-model"
    assert stub.kwargs["tools"] == [
        {"name": "t", "description": "d", "input_schema": {"type": "object"}}
    ]


def test_cost_is_tokens_over_1m_times_price() -> None:
    from agent_harness.llm import PRICE_PER_1M_INPUT, PRICE_PER_1M_OUTPUT

    assert cost_usd(1_000_000, 0) == pytest.approx(PRICE_PER_1M_INPUT)
    assert cost_usd(0, 1_000_000) == pytest.approx(PRICE_PER_1M_OUTPUT)
    assert cost_usd(0, 0) == 0.0
