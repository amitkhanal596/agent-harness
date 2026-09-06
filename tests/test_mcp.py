"""§3.4: MCP adapters — consuming a remote server's tools and exposing our own."""

from __future__ import annotations

from typing import Any

import pytest

from agent_harness.agent import Agent
from agent_harness.executor import Executor
from agent_harness.mcp import (
    MCPClient,
    MCPError,
    MCPToolAdapter,
    handle_mcp_call,
    mcp_tool_manifest,
    register_mcp_tools,
    specs_from_manifest,
)
from agent_harness.models import Completion, State, ToolCall
from agent_harness.tools import build_registry
from agent_harness.tools.base import Registry
from helpers import MockLLMClient

WEATHER_DESCRIPTOR = {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "inputSchema": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
        "required": ["city"],
    },
}


class _StubHTTP:
    """Stands in for httpx.AsyncClient, recording requests and replaying responses."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self.index = 0

    async def post(self, url: str, json: dict[str, Any],
                   headers: dict[str, str]) -> Any:
        self.requests.append(json)
        payload = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1

        class _Resp:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return payload

        return _Resp()

    async def __aenter__(self) -> _StubHTTP:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _patch(monkeypatch: pytest.MonkeyPatch, responses: list[dict[str, Any]]) -> _StubHTTP:
    stub = _StubHTTP(responses)
    monkeypatch.setattr("agent_harness.mcp.httpx.AsyncClient", lambda **kw: stub)
    return stub


# --- consuming ------------------------------------------------------------------------


async def test_list_tools_parses_the_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [{"jsonrpc": "2.0", "id": 1,
                          "result": {"tools": [WEATHER_DESCRIPTOR]}}])
    tools = await MCPClient("http://mcp.example/rpc").list_tools()
    assert tools[0]["name"] == "get_weather"


async def test_jsonrpc_envelope_is_well_formed(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _patch(monkeypatch, [{"result": {"tools": []}}])
    await MCPClient("http://mcp.example/rpc").list_tools()
    sent = stub.requests[0]
    assert sent["jsonrpc"] == "2.0"
    assert sent["method"] == "tools/list"
    assert isinstance(sent["id"], int)


async def test_server_error_becomes_an_mcp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [{"error": {"code": -32601, "message": "method not found"}}])
    with pytest.raises(MCPError, match="method not found"):
        await MCPClient("http://mcp.example/rpc").list_tools()


async def test_call_tool_flattens_content_blocks(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, [{"result": {"content": [
        {"type": "text", "text": "18C"}, {"type": "text", "text": "cloudy"}
    ]}}])
    out = await MCPClient("http://x/rpc").call_tool("get_weather", {"city": "Oslo"})
    assert out == "18C\ncloudy"


async def test_is_error_result_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error result must raise so the executor reports it as a tool error."""
    _patch(monkeypatch, [{"result": {"content": [{"type": "text", "text": "no such city"}],
                                     "isError": True}}])
    with pytest.raises(MCPError, match="no such city"):
        await MCPClient("http://x/rpc").call_tool("get_weather", {"city": "Nowhere"})


def test_adapter_derives_a_validating_input_model() -> None:
    adapter = MCPToolAdapter(MCPClient("http://x/rpc"), WEATHER_DESCRIPTOR)
    assert adapter.name == "get_weather"
    # Required field enforced...
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        adapter.InputModel.model_validate({})
    # ...optional one is not.
    assert adapter.InputModel.model_validate({"city": "Oslo"})


def test_adapter_passes_through_the_servers_own_schema() -> None:
    """The remote schema may carry constraints the model conversion cannot express."""
    adapter = MCPToolAdapter(MCPClient("http://x/rpc"), WEATHER_DESCRIPTOR)
    assert adapter.schema() == WEATHER_DESCRIPTOR["inputSchema"]


def test_adapter_handles_a_no_argument_tool() -> None:
    adapter = MCPToolAdapter(
        MCPClient("http://x/rpc"),
        {"name": "ping", "description": "p", "inputSchema": {"type": "object"}},
    )
    assert adapter.InputModel.model_validate({}) is not None


async def test_register_mcp_tools_adds_them_to_the_registry(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, [{"result": {"tools": [WEATHER_DESCRIPTOR]}}])
    registry = build_registry()
    names = await register_mcp_tools(registry, "http://mcp.example/rpc")
    assert names == ["get_weather"]
    assert registry.get("get_weather") is not None
    assert "get_weather" in {s.name for s in registry.specs()}


async def test_the_loop_uses_a_remote_mcp_tool_transparently(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of §3.4: the harness cannot tell a remote tool from a local one."""
    _patch(
        monkeypatch,
        [
            {"result": {"tools": [WEATHER_DESCRIPTOR]}},
            {"result": {"content": [{"type": "text", "text": "18C and cloudy"}]}},
        ],
    )
    registry = Registry()
    await register_mcp_tools(registry, "http://mcp.example/rpc")

    llm = MockLLMClient(
        [
            Completion(
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="c1", name="get_weather",
                                     input={"city": "Oslo"})],
            ),
            Completion(stop_reason="end_turn", text="It is 18C and cloudy in Oslo."),
        ]
    )
    agent = Agent(
        llm=llm,
        registry=registry,
        executor=Executor(registry, timeout=5.0, max_retries=0),
        max_steps=5,
    )
    from uuid import uuid4

    state = State(run_id=uuid4(), task="weather in Oslo")
    assert await agent.run(state) == "It is 18C and cloudy in Oslo."
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert "18C and cloudy" in tool_msg.content


async def test_invalid_arguments_to_a_remote_tool_are_caught_locally(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation happens before dispatch, so a bad call never leaves the process."""
    stub = _patch(monkeypatch, [{"result": {"tools": [WEATHER_DESCRIPTOR]}}])
    registry = Registry()
    await register_mcp_tools(registry, "http://mcp.example/rpc")
    calls_before = len(stub.requests)

    result = await Executor(registry, timeout=5.0, max_retries=0).run(
        ToolCall(id="c1", name="get_weather", input={})  # missing required 'city'
    )
    assert result.error is not None
    assert "invalid input" in result.error
    assert len(stub.requests) == calls_before  # no network call was made


# --- exposing -------------------------------------------------------------------------


def test_manifest_renders_our_registry_in_mcp_shapes() -> None:
    manifest = mcp_tool_manifest(build_registry())
    names = {t["name"] for t in manifest["tools"]}
    assert {"read_file", "grep", "run_shell"} <= names
    for tool in manifest["tools"]:
        assert "inputSchema" in tool
        assert "description" in tool


def test_manifest_round_trips_to_specs() -> None:
    specs = specs_from_manifest(mcp_tool_manifest(build_registry()))
    assert {s.name for s in specs} == {s.name for s in build_registry().specs()}


async def test_handle_call_executes_a_local_tool(temp_repo: object) -> None:
    from pathlib import Path

    assert isinstance(temp_repo, Path)
    result = await handle_mcp_call(
        build_registry(), "read_file", {"path": str(temp_repo / "config.txt")}
    )
    assert result["isError"] is False
    assert "port=8080" in result["content"][0]["text"]


async def test_handle_call_reports_unknown_tools_as_error_results() -> None:
    result = await handle_mcp_call(build_registry(), "nope", {})
    assert result["isError"] is True
    assert "unknown tool" in result["content"][0]["text"]


async def test_handle_call_reports_failures_as_error_results_not_raises() -> None:
    """The protocol expects an isError result, mirroring errors-as-data in the loop."""
    result = await handle_mcp_call(build_registry(), "read_file", {"path": "/no/such"})
    assert result["isError"] is True
    assert "content" in result


async def test_handle_call_validates_arguments() -> None:
    result = await handle_mcp_call(build_registry(), "read_file", {"wrong": 1})
    assert result["isError"] is True
