"""Tool package: `build_registry()` returns the fully populated Registry."""

from __future__ import annotations

from ..embeddings import VectorStore
from .base import Registry, Tool
from .grep import GrepTool
from .http_get import HttpGetTool
from .list_dir import ListDirTool
from .read_file import ReadFileTool
from .run_shell import RunShellTool
from .search_docs import SearchDocsTool

__all__ = ["Registry", "Tool", "build_registry"]


def build_registry(
    shell_timeout: float = 30.0,
    http_timeout: float = 20.0,
    vector_store: VectorStore | None = None,
) -> Registry:
    """Instantiate and register every built-in tool.

    `search_docs` is registered only when a `VectorStore` is supplied, so a deployment
    without pgvector simply does not advertise retrieval to the model rather than
    offering a tool that always errors.
    """
    registry = Registry()
    tools: list[Tool] = [
        ReadFileTool(),
        ListDirTool(),
        GrepTool(),
        RunShellTool(timeout=shell_timeout),
        HttpGetTool(timeout=http_timeout),
    ]
    if vector_store is not None:
        tools.append(SearchDocsTool(vector_store))
    for tool in tools:
        registry.register(tool)
    return registry
