"""M9: the local-model swap is one config value, and prompt caching is opt-in."""

from __future__ import annotations

from typing import Any

from agent_harness.config import Settings
from agent_harness.llm import AnthropicClient
from agent_harness.models import Message, ToolSpec


class _Block:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _Usage:
    input_tokens = 5
    output_tokens = 2


class _Response:
    content = [_Block(type="text", text="hi")]
    stop_reason = "end_turn"
    usage = _Usage()


class _StubMessages:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _Response:
        self.kwargs = kwargs
        return _Response()


def test_base_url_is_passed_through_to_the_sdk() -> None:
    client = AnthropicClient(
        model="local", api_key="sk-x", base_url="http://localhost:8080"
    )
    assert str(client._client.base_url).rstrip("/") == "http://localhost:8080"


def test_empty_base_url_falls_back_to_the_default_endpoint() -> None:
    default = AnthropicClient(model="m", api_key="sk-x")
    explicit_empty = AnthropicClient(model="m", api_key="sk-x", base_url="")
    assert str(default._client.base_url) == str(explicit_empty._client.base_url)
    assert "anthropic.com" in str(default._client.base_url)


def test_settings_expose_the_swap_as_one_value() -> None:
    settings = Settings(_env_file=None, anthropic_base_url="http://localhost:8080")  # type: ignore[call-arg]
    assert settings.anthropic_base_url == "http://localhost:8080"


async def test_prompt_caching_is_off_by_default() -> None:
    client = AnthropicClient(model="m", api_key="sk-x")
    stub = _StubMessages()
    client._client.messages = stub  # type: ignore[assignment]
    spec = ToolSpec(name="t", description="d", input_schema={"type": "object"})
    await client.complete([Message(role="user", content="hi")], [spec])
    assert "cache_control" not in str(stub.kwargs["tools"])


async def test_prompt_caching_marks_the_last_tool_when_enabled() -> None:
    """With caching on, the tool block carries a cache breakpoint (the stable prefix)."""
    client = AnthropicClient(model="m", api_key="sk-x", cache_prompt=True)
    stub = _StubMessages()
    client._client.messages = stub  # type: ignore[assignment]
    specs = [
        ToolSpec(name="a", description="d", input_schema={"type": "object"}),
        ToolSpec(name="b", description="d", input_schema={"type": "object"}),
    ]
    await client.complete([Message(role="user", content="hi")], specs)
    tools = stub.kwargs["tools"]
    assert "cache_control" not in tools[0]
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}


async def test_prompt_caching_with_no_tools_is_a_noop() -> None:
    client = AnthropicClient(model="m", api_key="sk-x", cache_prompt=True)
    stub = _StubMessages()
    client._client.messages = stub  # type: ignore[assignment]
    await client.complete([Message(role="user", content="hi")], [])
    assert stub.kwargs["tools"] == []
