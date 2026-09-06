"""M2: Registry lookup, tool schemas, and each tool's behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.tools import build_registry
from agent_harness.tools.base import Registry, Tool
from agent_harness.tools.grep import GrepInput, GrepTool
from agent_harness.tools.http_get import HttpGetInput, HttpGetTool
from agent_harness.tools.list_dir import ListDirInput, ListDirTool
from agent_harness.tools.read_file import ReadFileInput, ReadFileTool
from agent_harness.tools.run_shell import DisallowedCommandError, RunShellInput, RunShellTool


def test_build_registry_has_all_five_tools() -> None:
    registry = build_registry()
    names = {spec.name for spec in registry.specs()}
    assert names == {"read_file", "list_dir", "grep", "run_shell", "http_get"}


def test_get_returns_none_for_unknown_tool() -> None:
    assert build_registry().get("nope") is None


def test_specs_expose_json_schema_with_properties() -> None:
    spec = next(s for s in build_registry().specs() if s.name == "read_file")
    assert spec.input_schema["type"] == "object"
    assert "path" in spec.input_schema["properties"]
    assert spec.description


def test_register_replaces_same_name() -> None:
    registry = Registry()
    first, second = ReadFileTool(), ReadFileTool()
    registry.register(first)
    registry.register(second)
    assert registry.get("read_file") is second
    assert len(registry.specs()) == 1


def test_tool_is_abstract() -> None:
    with pytest.raises(TypeError):
        Tool()  # type: ignore[abstract]


async def test_read_file_returns_contents(temp_repo: Path) -> None:
    out = await ReadFileTool().execute(ReadFileInput(path=str(temp_repo / "config.txt")))
    assert "port=8080" in out


async def test_read_file_truncates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_harness.tools.read_file as rf

    big = tmp_path / "big.txt"
    big.write_text("x" * 5000)
    monkeypatch.setattr(rf, "MAX_BYTES", 100)
    out = await rf.ReadFileTool().execute(rf.ReadFileInput(path=str(big)))
    assert "[truncated at 100 bytes]" in out


async def test_read_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        await ReadFileTool().execute(ReadFileInput(path=str(tmp_path / "ghost.txt")))


async def test_list_dir_lists_entries(temp_repo: Path) -> None:
    out = await ListDirTool().execute(ListDirInput(path=str(temp_repo)))
    assert "config.txt" in out
    assert "sub/" in out


async def test_list_dir_on_file_raises(temp_repo: Path) -> None:
    with pytest.raises(NotADirectoryError):
        await ListDirTool().execute(ListDirInput(path=str(temp_repo / "config.txt")))


async def test_grep_file_returns_numbered_lines(temp_repo: Path) -> None:
    out = await GrepTool().execute(
        GrepInput(pattern=r"port=\d+", path=str(temp_repo / "config.txt"))
    )
    assert out.startswith("2:port=8080")


async def test_grep_directory_prefixes_filenames(temp_repo: Path) -> None:
    out = await GrepTool().execute(GrepInput(pattern="port", path=str(temp_repo)))
    assert "config.txt:2:port=8080" in out
    assert "deep.txt:1:port=9090" in out


async def test_grep_no_matches(temp_repo: Path) -> None:
    out = await GrepTool().execute(GrepInput(pattern="zzz-absent", path=str(temp_repo)))
    assert out == "[no matches]"


async def test_grep_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await GrepTool().execute(GrepInput(pattern="x", path=str(tmp_path / "ghost")))


async def test_run_shell_allowlisted_command() -> None:
    out = await RunShellTool().execute(RunShellInput(command="echo hello"))
    assert "hello" in out
    assert "exit_code=0" in out


async def test_run_shell_rejects_non_allowlisted() -> None:
    with pytest.raises(DisallowedCommandError):
        await RunShellTool().execute(RunShellInput(command="rm -rf /"))


async def test_run_shell_rejects_empty_command() -> None:
    with pytest.raises(DisallowedCommandError):
        await RunShellTool().execute(RunShellInput(command="   "))


async def test_run_shell_nonzero_exit_is_reported() -> None:
    out = await RunShellTool().execute(RunShellInput(command="cat /nonexistent-path-xyz"))
    assert "exit_code=1" in out


async def test_http_get_rejects_bad_scheme() -> None:
    with pytest.raises(ValueError):
        await HttpGetTool().execute(HttpGetInput(url="file:///etc/passwd"))
