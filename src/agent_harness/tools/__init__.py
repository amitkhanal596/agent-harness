"""Tool package: `build_registry()` returns the fully populated Registry."""

from __future__ import annotations

from .base import Registry, Tool
from .grep import GrepTool
from .http_get import HttpGetTool
from .list_dir import ListDirTool
from .read_file import ReadFileTool
from .run_shell import RunShellTool

__all__ = ["Registry", "Tool", "build_registry"]


def build_registry(shell_timeout: float = 30.0, http_timeout: float = 20.0) -> Registry:
    """Instantiate and register every built-in tool."""
    registry = Registry()
    for tool in (
        ReadFileTool(),
        ListDirTool(),
        GrepTool(),
        RunShellTool(timeout=shell_timeout),
        HttpGetTool(timeout=http_timeout),
    ):
        registry.register(tool)
    return registry
