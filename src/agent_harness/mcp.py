"""Model Context Protocol adapters: consume MCP servers' tools, and expose our own.

MCP is the emerging standard for how agents and tools find each other, so a harness with
a bespoke `Tool`/`Registry` interface and no MCP support is an island. Two directions,
both thin by design:

* **Consuming** — `MCPToolAdapter` wraps a tool living on a remote MCP server behind the
  existing `Tool` ABC, so the executor's validation, timeout, retry, and tracing apply to
  it exactly as they do to a built-in tool. Nothing else in the harness changes.
* **Exposing** — `mcp_tool_manifest` / `handle_mcp_call` render this harness's registry
  in MCP's shapes, so another client can drive these tools.

The JSON-RPC transport is implemented directly over `httpx` rather than taking the
official SDK as a dependency: the spec forbids agent frameworks, the wire format is a
handful of methods, and a thin adapter keeps the dependency surface honest.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, create_model

from .models import ToolSpec
from .tools.base import Registry, Tool


class MCPError(Exception):
    """The MCP server returned a JSON-RPC error or an unusable response."""


class MCPClient:
    """Minimal JSON-RPC client for an MCP server over HTTP.

    Implements only what tool use needs: `tools/list` and `tools/call`.
    """

    def __init__(self, url: str, timeout: float = 30.0, headers: dict[str, str] | None = None):
        self._url = url.rstrip("/")
        self._timeout = timeout
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._id = 0

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json=payload, headers=self._headers)
            resp.raise_for_status()
            body = resp.json()
        if "error" in body:
            error = body["error"]
            raise MCPError(f"{method} failed: {error.get('message', error)}")
        return body.get("result", {})

    async def list_tools(self) -> list[dict[str, Any]]:
        """Tool descriptors advertised by the server."""
        result = await self._call("tools/list")
        tools = result.get("tools", [])
        return list(tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a remote tool and flatten its content blocks to text.

        An `isError` result is raised rather than returned, so the executor converts it
        into `ToolResult.error` on the normal path instead of feeding the model an error
        string that looks like success.
        """
        result = await self._call("tools/call", {"name": name, "arguments": arguments})
        parts: list[str] = []
        for block in result.get("content", []):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(json.dumps(block))
        text = "\n".join(parts)
        if result.get("isError"):
            raise MCPError(text or f"tool {name!r} reported an error")
        return text


def _model_from_schema(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build a pydantic model from an MCP tool's JSON Schema.

    Only the shape needed to validate arguments before dispatch: property names,
    coarse types, and which are required. Anything unrecognised is left as `Any`, so a
    richer remote schema degrades to pass-through rather than rejecting valid calls.
    """
    types: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: dict[str, Any] = {}
    for field, spec in properties.items():
        python_type: Any = types.get(spec.get("type", ""), Any)
        fields[field] = (python_type, ... if field in required else None)
    if not fields:
        # A tool taking no arguments still needs a model the executor can validate against.
        return create_model(f"{name}_Input")
    return create_model(f"{name}_Input", **fields)


class MCPToolAdapter(Tool):
    """Presents a remote MCP tool as a local `Tool`.

    Because it satisfies the same ABC, the executor's schema validation, timeout, retry,
    and tracing all apply unchanged — the harness cannot tell it is remote.
    """

    def __init__(self, client: MCPClient, descriptor: dict[str, Any]) -> None:
        self._client = client
        self.name = descriptor["name"]
        self.description = descriptor.get("description", "")
        self._schema = descriptor.get("inputSchema") or {"type": "object", "properties": {}}
        self.InputModel = _model_from_schema(self.name, self._schema)

    def schema(self) -> dict[str, Any]:
        """The server's own schema, passed through verbatim.

        Preferred over the regenerated model's schema, which would lose constraints the
        conversion does not model.
        """
        return self._schema

    async def execute(self, args: BaseModel) -> str:
        """Dispatch to the remote server."""
        return await self._client.call_tool(
            self.name, args.model_dump(exclude_none=True)
        )


async def register_mcp_tools(
    registry: Registry, url: str, request_timeout: float = 30.0,
    headers: dict[str, str] | None = None,
) -> list[str]:
    """Discover an MCP server's tools and add them to `registry`. Returns their names."""
    client = MCPClient(url, timeout=request_timeout, headers=headers)
    names: list[str] = []
    for descriptor in await client.list_tools():
        registry.register(MCPToolAdapter(client, descriptor))
        names.append(descriptor["name"])
    return names


# --- exposing this harness's tools over MCP ------------------------------------------


def mcp_tool_manifest(registry: Registry) -> dict[str, Any]:
    """This registry rendered as an MCP `tools/list` result."""
    return {
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "inputSchema": spec.input_schema,
            }
            for spec in registry.specs()
        ]
    }


async def handle_mcp_call(
    registry: Registry, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Execute a local tool and render the outcome as an MCP `tools/call` result.

    Failures come back as `isError` results rather than exceptions, which is what the
    protocol expects and mirrors how the loop treats tool errors as data.
    """
    tool = registry.get(name)
    if tool is None:
        return {
            "content": [{"type": "text", "text": f"unknown tool: {name!r}"}],
            "isError": True,
        }
    try:
        validated = tool.InputModel.model_validate(arguments)
        output = await tool.execute(validated)
    except Exception as exc:  # noqa: BLE001 - protocol requires an error result, not a raise
        return {
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            "isError": True,
        }
    return {"content": [{"type": "text", "text": output}], "isError": False}


def specs_from_manifest(manifest: dict[str, Any]) -> list[ToolSpec]:
    """Convert an MCP manifest into internal `ToolSpec`s (useful for inspection)."""
    return [
        ToolSpec(
            name=t["name"],
            description=t.get("description", ""),
            input_schema=t.get("inputSchema", {}),
        )
        for t in manifest.get("tools", [])
    ]
