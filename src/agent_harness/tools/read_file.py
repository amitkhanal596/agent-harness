"""read_file: return a file's contents, truncated to a size cap."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool

MAX_BYTES = 64 * 1024


class ReadFileInput(BaseModel):
    path: str = Field(description="Path to the file to read.")


def _read(path: str, limit: int) -> tuple[str, bool]:
    data = Path(path).read_bytes()
    truncated = len(data) > limit
    return data[:limit].decode("utf-8", errors="replace"), truncated


class ReadFileTool(Tool):
    """Read a UTF-8 text file. Raises OSError if the path is missing or unreadable."""

    name = "read_file"
    description = (
        f"Read the contents of a text file. Output is truncated at {MAX_BYTES} bytes."
    )
    InputModel = ReadFileInput

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, ReadFileInput)
        text, truncated = await asyncio.to_thread(_read, args.path, MAX_BYTES)
        if truncated:
            text += f"\n\n[truncated at {MAX_BYTES} bytes]"
        return text
