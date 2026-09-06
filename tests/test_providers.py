"""§2.8 a second provider, and §2.9 streaming."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_harness.agent import Agent
from agent_harness.executor import Executor
from agent_harness.llm import AnthropicClient
from agent_harness.models import Completion, Message, State, ToolCall, ToolSpec
from agent_harness.providers import (
    OpenAIClient,
    openai_cost_usd,
    to_openai_messages,
    to_openai_tools,
)
from agent_harness.tools import build_registry

# --- OpenAI message mapping ----------------------------------------------------------


def test_user_message_maps_to_plain_content() -> None:
    assert to_openai_messages([Message(role="user", content="hi")]) == [
        {"role": "user", "content": "hi"}
    ]


def test_tool_role_stays_a_tool_message_unlike_anthropic() -> None:
    """The key dialect difference: OpenAI has a real `tool` role."""
    out = to_openai_messages(
        [Message(role="tool", tool_call_id="call_1", content="result text")]
    )
    assert out == [{"role": "tool", "tool_call_id": "call_1", "content": "result text"}]


def test_consecutive_tool_results_are_not_grouped() -> None:
    """Anthropic groups these into one user message; OpenAI must not."""
    out = to_openai_messages(
        [
            Message(role="tool", tool_call_id="a", content="A"),
            Message(role="tool", tool_call_id="b", content="B"),
        ]
    )
    assert len(out) == 2
    assert [m["tool_call_id"] for m in out] == ["a", "b"]


def test_assistant_tool_calls_use_the_function_envelope() -> None:
    out = to_openai_messages(
        [
            Message(
                role="assistant",
                content="looking",
                tool_calls=[ToolCall(id="c1", name="read_file", input={"path": "a.txt"})],
            )
        ]
    )
    call = out[0]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "read_file"
    # Arguments are a JSON *string* in this dialect, not a dict.
    assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}


def test_tools_are_wrapped_in_a_function_object() -> None:
    spec = ToolSpec(name="t", description="d", input_schema={"type": "object"})
    assert to_openai_tools([spec]) == [
        {
            "type": "function",
            "function": {"name": "t", "description": "d",
                         "parameters": {"type": "object"}},
        }
    ]


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValueError):
        to_openai_messages([Message(role="system", content="x")])


# --- OpenAI response parsing ---------------------------------------------------------


class _StubTransport:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self.sent: dict[str, Any] = {}

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> Any:
        self.sent = json

        class _Resp:
            def __init__(self, payload: dict[str, Any]) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return self._payload

        return _Resp(self.body)

    async def __aenter__(self) -> _StubTransport:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _patch(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> _StubTransport:
    stub = _StubTransport(body)
    monkeypatch.setattr(
        "agent_harness.providers.httpx.AsyncClient", lambda **kw: stub
    )
    return stub


async def test_openai_tool_call_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(
        monkeypatch,
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "x.txt"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7},
        },
    )
    client = OpenAIClient(model="gpt-4o", api_key="sk-test")
    comp = await client.complete([Message(role="user", content="hi")], [])

    assert comp.tool_calls == [
        ToolCall(id="call_abc", name="read_file", input={"path": "x.txt"})
    ]
    # Finish reason is normalised into the internal vocabulary.
    assert comp.stop_reason == "tool_use"
    assert (comp.prompt_tokens, comp.output_tokens) == (42, 7)


async def test_openai_text_response_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(
        monkeypatch,
        {
            "choices": [{"message": {"content": "the answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        },
    )
    comp = await OpenAIClient(model="gpt-4o", api_key="sk-test").complete(
        [Message(role="user", content="hi")], []
    )
    assert comp.text == "the answer"
    assert comp.stop_reason == "end_turn"
    assert comp.tool_calls == []


async def test_malformed_tool_arguments_do_not_crash(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad JSON string becomes an empty dict, so schema validation reports it."""
    _patch(
        monkeypatch,
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "read_file", "arguments": "{not json"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        },
    )
    comp = await OpenAIClient(model="gpt-4o", api_key="sk-test").complete([], [])
    assert comp.tool_calls[0].input == {}


async def test_base_url_enables_compatible_servers(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """llama.cpp / vLLM / Ollama are reached by pointing base_url at them."""
    stub = _patch(
        monkeypatch,
        {"choices": [{"message": {"content": "local"}, "finish_reason": "stop"}],
         "usage": {}},
    )
    client = OpenAIClient(model="llama-3", api_key="none",
                          base_url="http://localhost:8080/v1")
    comp = await client.complete([Message(role="user", content="hi")], [])
    assert comp.text == "local"
    assert stub.sent["model"] == "llama-3"


def test_openai_cost_uses_its_own_pricing() -> None:
    from agent_harness.providers import (
        OPENAI_PRICE_PER_1M_INPUT,
        OPENAI_PRICE_PER_1M_OUTPUT,
    )

    assert openai_cost_usd(1_000_000, 0) == pytest.approx(OPENAI_PRICE_PER_1M_INPUT)
    assert openai_cost_usd(0, 1_000_000) == pytest.approx(OPENAI_PRICE_PER_1M_OUTPUT)


# --- the abstraction actually holds ---------------------------------------------------


async def test_the_loop_runs_unchanged_on_the_openai_client(
    monkeypatch: pytest.MonkeyPatch, temp_repo: object
) -> None:
    """The point of §2.8: swapping providers requires no change to the loop."""
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "list_dir",
                                             "arguments": '{"path": "."}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        {
            "choices": [{"message": {"content": "listed it"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    ]

    class _Sequenced(_StubTransport):
        def __init__(self) -> None:
            super().__init__({})
            self.index = 0

        async def post(self, url: str, headers: dict[str, str],
                       json: dict[str, Any]) -> Any:
            self.body = responses[min(self.index, len(responses) - 1)]
            self.index += 1
            return await super().post(url, headers, json)

    stub = _Sequenced()
    monkeypatch.setattr("agent_harness.providers.httpx.AsyncClient", lambda **kw: stub)

    registry = build_registry()
    agent = Agent(
        llm=OpenAIClient(model="gpt-4o", api_key="sk-test"),
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        max_steps=5,
    )
    from uuid import uuid4

    state = State(run_id=uuid4(), task="list the directory")
    assert await agent.run(state) == "listed it"
    assert state.status == "completed"
    # A tool result came back through the OpenAI `tool` role.
    assert any(m.role == "tool" for m in state.messages)


# --- streaming (§2.9) -----------------------------------------------------------------


class _StreamCtx:
    def __init__(self, chunks: list[str], final: Any) -> None:
        self._chunks = chunks
        self._final = final

    async def __aenter__(self) -> Any:
        chunks = self._chunks
        final = self._final

        class _S:
            @property
            def text_stream(self) -> Any:
                async def gen() -> Any:
                    for c in chunks:
                        yield c

                return gen()

            async def get_final_message(self) -> Any:
                return final

        return _S()

    async def __aexit__(self, *exc: object) -> None:
        return None


class _Block:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _Usage:
    def __init__(self, i: int, o: int) -> None:
        self.input_tokens, self.output_tokens = i, o


class _Final:
    def __init__(self, content: list[Any], stop_reason: str, usage: _Usage) -> None:
        self.content, self.stop_reason, self.usage = content, stop_reason, usage


async def test_streaming_yields_deltas_and_assembles_the_completion() -> None:
    client = AnthropicClient(model="m", api_key="sk-test")
    final = _Final(
        [_Block(type="text", text="hello world")], "end_turn", _Usage(10, 4)
    )

    class _Messages:
        def stream(self, **kwargs: Any) -> _StreamCtx:
            return _StreamCtx(["hello ", "world"], final)

    client._client.messages = _Messages()  # type: ignore[assignment]

    received: list[str] = []
    async with client.stream([Message(role="user", content="hi")], []) as deltas:
        async for chunk in deltas:
            received.append(chunk)

    assert received == ["hello ", "world"]
    assert client.last_stream is not None
    assert client.last_stream.text == "hello world"
    assert client.last_stream.prompt_tokens == 10
    assert client.last_stream.output_tokens == 4


async def test_streaming_assembles_tool_calls_only_when_complete() -> None:
    """A half-received tool_use must never be surfaced; the loop would act on garbage."""
    client = AnthropicClient(model="m", api_key="sk-test")
    final = _Final(
        [
            _Block(type="text", text="let me look"),
            _Block(type="tool_use", id="tu_1", name="read_file",
                   input={"path": "a.txt"}),
        ],
        "tool_use",
        _Usage(20, 8),
    )

    class _Messages:
        def stream(self, **kwargs: Any) -> _StreamCtx:
            return _StreamCtx(["let ", "me ", "look"], final)

    client._client.messages = _Messages()  # type: ignore[assignment]

    async with client.stream([Message(role="user", content="hi")], []) as deltas:
        chunks = [c async for c in deltas]

    assert chunks == ["let ", "me ", "look"]
    assert client.last_stream is not None
    assert client.last_stream.tool_calls == [
        ToolCall(id="tu_1", name="read_file", input={"path": "a.txt"})
    ]
    assert client.last_stream.stop_reason == "tool_use"


async def test_completion_from_a_stream_is_loop_compatible() -> None:
    """The assembled Completion is the same shape the loop consumes from `complete`."""
    client = AnthropicClient(model="m", api_key="sk-test")
    final = _Final([_Block(type="text", text="done")], "end_turn", _Usage(1, 1))

    class _Messages:
        def stream(self, **kwargs: Any) -> _StreamCtx:
            return _StreamCtx(["done"], final)

    client._client.messages = _Messages()  # type: ignore[assignment]
    async with client.stream([], []) as deltas:
        async for _ in deltas:
            pass

    assert isinstance(client.last_stream, Completion)
