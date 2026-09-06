"""§2.5: the run_shell allowlist must resist interpreter and --exec style escapes."""

from __future__ import annotations

import pytest

from agent_harness.tools.run_shell import (
    ALLOWED_COMMANDS,
    DisallowedCommandError,
    RunShellInput,
    RunShellTool,
    check_command,
)


@pytest.mark.parametrize(
    "command",
    [
        # Interpreters are the classic argv[0]-only allowlist bypass.
        'python -c "import os; os.system(\'echo pwned\')"',
        'python3 -c "print(1)"',
        "sh -c 'echo pwned'",
        "bash -c 'echo pwned'",
        "perl -e 'print 1'",
        "ruby -e 'puts 1'",
        "node -e 'console.log(1)'",
        "zsh -c 'echo pwned'",
        # Not allowlisted at all.
        "rm -rf /",
        "curl http://example.com",
        "chmod 777 /etc/passwd",
        "nc -l 4444",
    ],
)
def test_interpreters_and_unlisted_programs_are_rejected(command: str) -> None:
    import shlex

    with pytest.raises(DisallowedCommandError):
        check_command(shlex.split(command))


@pytest.mark.parametrize(
    "command",
    [
        # Flags that let an allowlisted program execute something else.
        "find . -name x -exec rm {} ;",
        "find . -execdir sh {} ;",
        "find . -delete",
        "git -c core.pager=sh log",
        "git --exec-path=/tmp/evil status",
        "git commit -m x",
        "git push origin main",
        "git config --global core.editor sh",
        "awk -f /tmp/evil.awk",
        "sed -i s/a/b/ file.txt",
        "pytest -p evil_plugin",
        "pytest --pdb",
    ],
)
def test_escape_flags_on_allowlisted_programs_are_rejected(command: str) -> None:
    import shlex

    with pytest.raises(DisallowedCommandError):
        check_command(shlex.split(command))


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "cat README.md",
        "grep -rn TODO .",
        "find . -name '*.py'",
        "git log --oneline",
        "git status",
        "git diff",
        "pytest -q",
        "wc -l file.txt",
        "sed s/a/b/ file.txt",
        "head -20 file.txt",
    ],
)
def test_legitimate_read_only_commands_are_allowed(command: str) -> None:
    import shlex

    check_command(shlex.split(command))  # must not raise


def test_path_prefixed_program_is_still_checked() -> None:
    """An absolute path must not smuggle a non-allowlisted program past the check."""
    with pytest.raises(DisallowedCommandError):
        check_command(["/usr/bin/python3", "-c", "print(1)"])
    check_command(["/bin/ls", "-la"])  # allowlisted basename is fine


def test_flag_with_attached_value_is_caught() -> None:
    """`--exec-path=/x` must be caught the same as `--exec-path /x`."""
    with pytest.raises(DisallowedCommandError):
        check_command(["git", "--exec-path=/tmp/evil", "status"])


def test_empty_command_rejected() -> None:
    with pytest.raises(DisallowedCommandError):
        check_command([])


def test_no_interpreter_is_on_the_allowlist() -> None:
    """Regression guard: nobody re-adds a general-purpose interpreter."""
    interpreters = {"python", "python3", "sh", "bash", "zsh", "perl", "ruby", "node", "env"}
    assert not (interpreters & set(ALLOWED_COMMANDS))


async def test_disallowed_command_surfaces_as_a_tool_error_not_a_crash() -> None:
    """The executor turns this into ToolResult.error; here we prove the tool raises."""
    with pytest.raises(DisallowedCommandError):
        await RunShellTool().execute(RunShellInput(command="python3 -c 'print(1)'"))


async def test_allowed_command_still_runs() -> None:
    out = await RunShellTool().execute(RunShellInput(command="echo hello"))
    assert "hello" in out
    assert "exit_code=0" in out
