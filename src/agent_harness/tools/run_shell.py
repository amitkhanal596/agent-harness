"""run_shell: execute an allowlisted command, capturing stdout and stderr."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import PurePath

from pydantic import BaseModel, Field

from .base import Tool

#: Programs the agent may run, mapped to the flags/subcommands each one forbids.
#:
#: Allowlisting the *program* is not enough on its own: an interpreter or a tool with an
#: `--exec`-style escape hatch turns a one-entry allowlist into arbitrary code execution.
#: `python -c "import os; os.system(...)"` passes any argv[0]-only check trivially. So
#: each entry also names the argument patterns that re-open that hole.
#:
#: Interpreters (`python`, `python3`, `sh`, `bash`, `perl`, `ruby`, `node`) are absent by
#: design — there is no subset of their flags that is safely lockable, and no built-in
#: tool or eval task needs them. `pytest` is the one interpreter-adjacent entry kept,
#: because CI triage is the harness's headline task; its plugin-loading escape (`-p`)
#: is blocked below.
ALLOWED_COMMANDS: dict[str, frozenset[str]] = {
    "ls": frozenset(),
    "cat": frozenset(),
    "head": frozenset(),
    "tail": frozenset(),
    "wc": frozenset(),
    "echo": frozenset(),
    "pwd": frozenset(),
    "grep": frozenset(),
    "sort": frozenset(),
    "uniq": frozenset(),
    "diff": frozenset(),
    "stat": frozenset(),
    "which": frozenset(),
    # `find -exec/-execdir/-delete/-fprintf` run programs or write files.
    "find": frozenset({"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprintf", "-fls"}),
    # `sed -e/-f` with the `e` command, and `-i`, execute or rewrite. Keep it read-only.
    "sed": frozenset({"-i", "--in-place", "-s", "--separate"}),
    # `awk` can shell out via system()/print|"cmd"; `-f` loads an arbitrary program file.
    "awk": frozenset({"-f", "--file", "--source", "-v"}),
    # git subcommands that run hooks, filters, or user-supplied programs.
    "git": frozenset(
        {
            "-c",
            "--exec-path",
            "-C",
            "--upload-pack",
            "--receive-pack",
            "commit",
            "push",
            "pull",
            "fetch",
            "clone",
            "filter-branch",
            "submodule",
            "daemon",
            "send-email",
            "gc",
            "config",
            "clean",
            "reset",
            "checkout",
            "rebase",
            "merge",
            "apply",
            "am",
        }
    ),
    # `pytest -p` loads arbitrary plugins; the rest of pytest is a test runner.
    "pytest": frozenset({"-p", "--plugin", "--pdb", "--pdbcls", "-c", "--rootdir"}),
}

DEFAULT_TIMEOUT = 30.0
MAX_OUTPUT_BYTES = 32 * 1024


class RunShellInput(BaseModel):
    command: str = Field(description="Command line to run. The program must be allowlisted.")


class DisallowedCommandError(Exception):
    """Raised when the requested program or one of its arguments is not permitted."""


def check_command(parts: list[str]) -> None:
    """Validate a split command line against the allowlist.

    Raises `DisallowedCommandError` when the program is not allowlisted or when one of
    its arguments matches that program's forbidden set. Checking arguments matters as
    much as checking the program: several allowlisted tools can execute other programs
    through a flag.
    """
    if not parts:
        raise DisallowedCommandError("empty command")

    program = PurePath(parts[0]).name
    if program not in ALLOWED_COMMANDS:
        raise DisallowedCommandError(
            f"command {program!r} is not allowlisted; allowed: "
            f"{', '.join(sorted(ALLOWED_COMMANDS))}"
        )

    forbidden = ALLOWED_COMMANDS[program]
    for arg in parts[1:]:
        # Compare the flag itself, so `--exec-path=/x` is caught alongside `--exec-path`.
        token = arg.split("=", 1)[0]
        if token in forbidden:
            raise DisallowedCommandError(
                f"argument {token!r} is not permitted for {program!r}: it can execute "
                "another program or write to the filesystem"
            )


class RunShellTool(Tool):
    """Run a command without a shell.

    Raises DisallowedCommandError when the program or an argument is not permitted, and
    TimeoutError if the process outruns the timeout; both surface to the model as a tool
    error rather than crashing the loop.
    """

    name = "run_shell"
    description = (
        "Run a command and return its combined stdout and stderr. Only these programs "
        f"are permitted: {', '.join(sorted(ALLOWED_COMMANDS))}. Flags that would let a "
        "permitted program execute another program are rejected. Shell metacharacters "
        "(pipes, redirects, ;, &&) are not interpreted."
    )
    InputModel = RunShellInput

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    async def execute(self, args: BaseModel) -> str:
        assert isinstance(args, RunShellInput)
        parts = shlex.split(args.command)
        check_command(parts)

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
