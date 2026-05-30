"""run_command — execute an allowlisted external command from the agent.

The tool takes `argv` as a list (never a shell string) and uses
``asyncio.create_subprocess_exec`` so there is no shell parsing — no
injection risk through the argv path. The allowlist matches against
``Path(argv[0]).name`` so absolute paths and bare command names are
treated equivalently.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path

from .tool import Tool, ToolKind
from .types import ToolDef


# Argument-level escapes that let an allowlisted command launch *another*
# command or destroy files, bypassing the argv[0] allowlist. `find -exec`,
# `find -delete`, `xargs`-style fan-out, etc. Matched case-sensitively as
# exact argv tokens (these are literal flag spellings).
DEFAULT_FORBIDDEN_ARGS = frozenset({
    "-exec", "-execdir", "-ok", "-okdir", "-delete",
    "-fprint", "-fprintf", "-fprint0",
})

# Path globs that must never be passed to run_command — secrets that a
# hijacked agent could read (`cat`) and then be coaxed into exfiltrating.
# Matched (case-insensitively) against each argv token's raw, ~-expanded, and
# fully-resolved forms, and against its basename. Tighten/extend via the
# CommandPolicy.denied_path_patterns field.
DEFAULT_DENIED_PATHS = frozenset({
    "*/.env", "*.env", "*/.env.*",          # dotenv (incl. .env.local etc.)
    "*/.ssh/*", "*/id_rsa*", "*/id_ed25519*", "*/id_ecdsa*", "*/id_dsa*",
    "*/.aws/*", "*/.gnupg/*",
    "*/.netrc", "*/_netrc",
    "*/.config/gh/*", "*/.config/gcloud/*",
    "*/.ambi/.env",
})

# Suffixes that look like a secret path but are conventionally safe templates
# (committed, no real secrets) — exempt so the assistant can still read them.
_SAFE_SECRET_SUFFIXES = (".example", ".sample", ".template", ".dist")


def _is_denied_path(token: str, cwd: Path | None, patterns: frozenset[str]) -> bool:
    base = os.path.basename(token.rstrip("/"))
    if base.lower().endswith(_SAFE_SECRET_SUFFIXES):
        return False
    expanded = os.path.expanduser(token)
    forms = {token, expanded}
    try:
        root = cwd if cwd is not None else Path.cwd()
        forms.add(str((root / expanded).resolve()))
        forms.add(str(Path(expanded).resolve()))
    except (OSError, ValueError, RuntimeError):
        pass
    for form in forms:
        norm = form.replace("\\", "/").lower()
        cand = (norm, os.path.basename(norm))
        for pat in patterns:
            pl = pat.lower()
            if any(fnmatch.fnmatch(c, pl) for c in cand):
                return True
    return False


@dataclass
class CommandPolicy:
    """Policy controlling which commands run_command may execute."""

    allowed: set[str] = field(default_factory=set)
    cwd_root: Path | None = None
    default_timeout: float = 30.0
    max_output_bytes: int = 100_000
    forbidden_args: frozenset[str] = DEFAULT_FORBIDDEN_ARGS
    denied_path_patterns: frozenset[str] = DEFAULT_DENIED_PATHS


def make_run_command_tool(
    policy: CommandPolicy, kind: ToolKind = "write"
) -> Tool:
    """Build the `run_command` tool bound to a CommandPolicy.

    Defaults to ``kind="write"`` because the allowlist may include commands
    that mutate state (e.g. ``git push``). Pass ``kind="read"`` if your
    allowlist is strictly read-only.
    """

    async def handler(args: dict, progress=None) -> str:
        if progress is None:
            async def _noop(_msg: str) -> None:
                pass
            progress = _noop

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

        # Block argument-level escapes (e.g. `find … -exec sh -c …`) that would
        # spawn an arbitrary, un-allowlisted command. Cheap second layer behind
        # the allowlist itself.
        forbidden = policy.forbidden_args & set(argv[1:])
        if forbidden:
            return (
                f"Error: argument(s) {sorted(forbidden)} are forbidden — they "
                f"can launch un-allowlisted commands or delete files."
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

        # Block reads of sensitive paths (e.g. `cat .env`, `grep x ~/.ssh/id_rsa`).
        # Resolution is cwd-relative so relative paths are caught too.
        if policy.denied_path_patterns:
            for token in argv[1:]:
                if _is_denied_path(token, cwd_path, policy.denied_path_patterns):
                    return (
                        f"Error: access to sensitive path '{token}' is blocked "
                        f"by policy."
                    )

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

        stdout_buf: list[bytes] = []
        stderr_buf: list[bytes] = []

        # Cap progress emissions so a chatty command (find, grep -r) can't
        # flood the transport. Beyond the cap we keep collecting output for
        # the final result but stop emitting per-line progress.
        emit_count = [0]
        dropped_count = [0]
        PROGRESS_CAP = 100

        async def emit_line(text: str) -> None:
            if emit_count[0] < PROGRESS_CAP:
                await progress(text)
                emit_count[0] += 1
            else:
                dropped_count[0] += 1

        async def read_stream(stream, buf: list[bytes], prefix: str = "") -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                buf.append(line)
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if text:
                    await emit_line(f"{prefix}{text}")

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    read_stream(proc.stdout, stdout_buf),
                    read_stream(proc.stderr, stderr_buf, prefix="stderr: "),
                    proc.wait(),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Error: command timed out after {timeout}s"

        if dropped_count[0] > 0:
            await progress(
                f"+ {dropped_count[0]} more lines collected (not emitted)"
            )

        stdout = _decode_truncate(b"".join(stdout_buf), policy.max_output_bytes)
        stderr = _decode_truncate(b"".join(stderr_buf), policy.max_output_bytes)
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
