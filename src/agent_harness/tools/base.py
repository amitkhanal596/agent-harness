"""Tool abstract base class and the name -> tool Registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from ..models import ToolSpec


class Tool(ABC):
    """One callable capability exposed to the model.

    Subclasses set `name`, `description`, and an `InputModel` whose fields become the
    advertised JSON schema. `execute` receives an already-validated model instance.
    """

    name: str
    description: str
    InputModel: type[BaseModel]

    def schema(self) -> dict[str, Any]:
        """JSON Schema for this tool's input, derived from `InputModel`."""
        return self.InputModel.model_json_schema()

    @abstractmethod
    async def execute(self, args: BaseModel) -> str:
        """Run the tool and return its output as text. Raises on failure."""
        ...


class Registry:
    """In-memory tool lookup and spec generation."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool, replacing any existing tool with the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return the tool called `name`, or None if it is not registered."""
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        """Every registered tool, in the shape the model expects."""
        return [
            ToolSpec(name=t.name, description=t.description, input_schema=t.schema())
            for t in self._tools.values()
        ]
