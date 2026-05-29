"""run_command — execute an allowlisted external command from the agent.

The tool takes `argv` as a list (never a shell string) and uses
``asyncio.create_subprocess_exec`` so there is no shell parsing — no
injection risk through the argv path. The allowlist matches against
``Path(argv[0]).name`` so absolute paths and bare command names are
treated equivalently.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from .tool import Tool, ToolKind
from .types import ToolDef


@dataclass
class CommandPolicy:
    """Policy controlling which commands run_command may execute."""

    allowed: set[str] = field(default_factory=set)
    cwd_root: Path | None = None
    default_timeout: float = 30.0
    max_output_bytes: int = 100_000


def make_run_command_tool(
    policy: CommandPolicy, kind: ToolKind = "write"
) -> Tool:
    """Build the `run_command` tool bound to a CommandPolicy.

    Defaults to ``kind="write"`` because the allowlist may include commands
    that mutate state (e.g. ``git push``). Pass ``kind="read"`` if your
    allowlist is strictly read-only.
    """

    async def handler(args: dict) -> str:
        argv = args.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(a, str) for a in argv)
        ):
            return "Error: argv must be a non-empty list of strings."

        cmd_name = Path(argv[0]).name
        if cmd_name not in policy.allowed:
            allowed_list = ", ".join(sorted(policy.allowed)) or "(none)"
            return (
                f"Error: command '{cmd_name}' is not allowlisted. "
                f"Allowed: {allowed_list}"
            )

        cwd_path: Path | None = None
        cwd_raw = args.get("cwd")
        if cwd_raw:
            cwd_path = Path(cwd_raw).resolve()
            if policy.cwd_root is not None:
                root = policy.cwd_root.resolve()
                try:
                    cwd_path.relative_to(root)
                except ValueError:
                    return f"Error: cwd '{cwd_raw}' must be under {root}"
            if not cwd_path.is_dir():
                return f"Error: cwd '{cwd_raw}' is not a directory."

        timeout = float(args.get("timeout") or policy.default_timeout)

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd_path) if cwd_path else None,
            )
        except FileNotFoundError:
            return f"Error: command '{argv[0]}' not found on PATH."

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Error: command timed out after {timeout}s"

        stdout = _decode_truncate(stdout_b, policy.max_output_bytes)
        stderr = _decode_truncate(stderr_b, policy.max_output_bytes)
        return (
            f"exit_code: {proc.returncode}\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}"
        )

    allowed_str = ", ".join(sorted(policy.allowed)) or "(none)"
    return Tool(
        definition=ToolDef(
            name="run_command",
            description=(
                f"Run an allowlisted shell command. Allowed: {allowed_str}. "
                "Pass argv as a list (e.g. ['git', 'status']) — never a "
                "single shell string. Returns exit_code, stdout, and stderr "
                "(truncated)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command as a list of strings, e.g. ['gh', 'pr', 'list']",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (optional).",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Per-invocation timeout in seconds.",
                    },
                },
                "required": ["argv"],
            },
        ),
        handler=handler,
        kind=kind,
    )


def _decode_truncate(data: bytes, limit: int) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, kept first {limit} bytes]"
