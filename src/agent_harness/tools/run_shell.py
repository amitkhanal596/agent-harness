"""run_shell: execute an allowlisted command, capturing stdout and stderr."""

from __future__ import annotations

import asyncio
import shlex

from pydantic import BaseModel, Field

from .base import Tool

# Only these executables may be run. Anything else is rejected before spawning.
ALLOWED_COMMANDS = frozenset(
    {"ls", "cat", "head", "tail", "wc", "echo", "pwd", "grep", "find", "git", "python",
     "python3", "pytest", "sed", "awk", "sort", "uniq", "diff", "stat", "which"}
)

DEFAULT_TIMEOUT = 30.0
MAX_OUTPUT_BYTES = 32 * 1024


class RunShellInput(BaseModel):
    command: str = Field(description="Command line to run. The program must be allowlisted.")


class DisallowedCommandError(Exception):
    """Raised when the requested program is not on the allowlist."""


class RunShellTool(Tool):
    """Run a command without a shell.

    Raises DisallowedCommandError for a non-allowlisted program and TimeoutError if the
    process outruns the timeout; both surface to the model as a tool error.
    """

    name = "run_shell"
    description = (
        "Run a shell command and return its combined stdout and stderr. Only these "
        f"programs are permitted: {', '.join(sorted(ALLOWED_COMMANDS))}. "
        "Shell metacharacters (pipes, redirects, ;, &&) are not interpreted."
    )
    InputModel = RunShellInput

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, RunShellInput)
        parts = shlex.split(args.command)
        if not parts:
            raise DisallowedCommandError("empty command")
        program = parts[0].rsplit("/", 1)[-1]
        if program not in ALLOWED_COMMANDS:
            raise DisallowedCommandError(
                f"command {program!r} is not allowlisted; allowed: "
                f"{', '.join(sorted(ALLOWED_COMMANDS))}"
            )

        proc = await asyncio.create_subprocess_exec(
            *parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"command timed out after {self._timeout}s") from None

        text = stdout[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        if len(stdout) > MAX_OUTPUT_BYTES:
            text += f"\n[truncated at {MAX_OUTPUT_BYTES} bytes]"
        return f"exit_code={proc.returncode}\n{text}" if text else f"exit_code={proc.returncode}"
