"""Obsidian vault integration — read/write notes as files in the vault.

The vault is a directory tree of markdown files. ambi reads/writes the
live filesystem so manually-authored notes and ambi-generated notes
coexist. All paths are validated to stay inside the vault root (no
``..`` escape).

Tools:
    obsidian_save(title, content, folder?, tags?, source?) -> str   [write]
    obsidian_list(folder?)                                  -> str  [read]
    obsidian_search(query, folder?)                         -> str  [read]
    obsidian_read(path)                                     -> str  [read]
    obsidian_delete(path)                                   -> str  [write]
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ...tool import Tool, ToolRegistry
from ...types import ToolDef


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


# ---------------------------------------------------------------------------
# Vault-root resolution + safety
# ---------------------------------------------------------------------------


class VaultError(ValueError):
    pass


def _resolve_under_vault(vault: Path, relative: str | Path) -> Path:
    """Return absolute path; raise if it escapes the vault."""
    candidate = (vault / relative).resolve()
    vault_resolved = vault.resolve()
    try:
        candidate.relative_to(vault_resolved)
    except ValueError:
        raise VaultError(f"path '{relative}' escapes vault root")
    return candidate


def _safe_title(title: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]", "_", title.strip())
    if not cleaned:
        raise VaultError("title is empty")
    return cleaned


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    body = text[m.end():]
    return fm, body


def _render_frontmatter(meta: dict) -> str:
    serialized = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{serialized}\n---\n"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _make_handlers(vault: Path, default_folder: str):
    async def save(args: dict) -> str:
        try:
            title = _safe_title(str(args.get("title") or ""))
        except VaultError as e:
            return f"Error: {e}"
        content = args.get("content")
        if not isinstance(content, str) or not content.strip():
            return "Error: 'content' (non-empty string) is required."
        # Default-to-Inbox: new notes land in the configured capture folder
        # unless the caller explicitly named one. Keeps the vault root clean
        # and matches PARA convention.
        folder = (args.get("folder") or default_folder).strip("/")
        tags_raw = args.get("tags")

        try:
            dest_dir = _resolve_under_vault(vault, folder) if folder else vault
            dest_dir.mkdir(parents=True, exist_ok=True)
        except VaultError as e:
            return f"Error: {e}"
        dest = dest_dir / f"{title}.md"

        meta: dict = {
            "title": title,
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if tags_raw:
            tags = [t.strip() for t in str(tags_raw).split(",") if t.strip()]
            if tags:
                meta["tags"] = tags
        if args.get("source"):
            meta["source"] = str(args["source"])

        text = _render_frontmatter(meta) + "\n" + content.rstrip() + "\n"
        dest.write_text(text, encoding="utf-8")
        rel = dest.relative_to(vault.resolve())
        return f"Saved: {rel} ({len(text)} bytes)"

    async def list_notes(args: dict) -> str:
        folder = (args.get("folder") or "").strip("/")
        root = _resolve_under_vault(vault, folder) if folder else vault.resolve()
        if not root.is_dir():
            return f"Error: '{folder or '.'}' is not a directory."

        paths = sorted(root.rglob("*.md"))

        # For huge result sets — common when listing a whole vault root — a
        # full path dump (a) overflows the agent's context window and (b)
        # isn't useful. Switch to a folder-breakdown summary that tells the
        # agent how to drill in.
        SUMMARY_THRESHOLD = 100
        if len(paths) > SUMMARY_THRESHOLD:
            files_at_root = 0
            subfolder_counts: dict[str, int] = {}
            for path in paths:
                rel = path.relative_to(root)
                if len(rel.parts) == 1:
                    files_at_root += 1
                else:
                    head = rel.parts[0]
                    subfolder_counts[head] = subfolder_counts.get(head, 0) + 1

            scope = f" under '{folder}'" if folder else ""
            lines = [
                f"{len(paths)} notes found{scope} (too many to list — showing folder breakdown):",
                "",
            ]
            if files_at_root:
                lines.append(f"  ./   {files_at_root} note(s) at the top of this scope")
            for name, count in sorted(
                subfolder_counts.items(), key=lambda x: (-x[1], x[0])
            ):
                sub = f"{folder}/{name}" if folder else name
                lines.append(f"  {sub}/   {count} note(s)")
            lines.append("")
            lines.append(
                'Use obsidian_list({"folder": "<path>"}) to drill into a '
                'specific folder, or obsidian_search({"query": "..."}) to '
                'find notes by content.'
            )
            return "\n".join(lines)

        if not paths:
            return "(no notes found)"

        rows: list[str] = []
        for path in paths:
            rel = path.relative_to(vault.resolve())
            text = _read_safe(path)
            meta, _ = _parse_frontmatter(text)
            title = meta.get("title") or path.stem
            tags = meta.get("tags") or []
            tag_str = f" [{','.join(str(t) for t in tags)}]" if tags else ""
            rows.append(f"{rel} — {title}{tag_str}")
        return "\n".join(rows)

    async def search(args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "Error: 'query' is required."
        folder = (args.get("folder") or "").strip("/")
        root = _resolve_under_vault(vault, folder) if folder else vault.resolve()
        if not root.is_dir():
            return f"Error: '{folder or '.'}' is not a directory."
        needle = query.lower()
        matches: list[str] = []
        for path in sorted(root.rglob("*.md")):
            rel = path.relative_to(vault.resolve())
            text = _read_safe(path)
            text_lc = text.lower()
            if needle in path.stem.lower() or needle in text_lc:
                snippet = _make_snippet(text, query)
                matches.append(f"{rel}\n    …{snippet}…")
        if not matches:
            return f"(no matches for '{query}')"
        return "\n".join(matches)

    async def read(args: dict) -> str:
        rel = args.get("path")
        if not rel:
            return "Error: 'path' is required."
        try:
            target = _resolve_under_vault(vault, rel)
        except VaultError as e:
            return f"Error: {e}"
        if not target.exists():
            return f"Error: '{rel}' not found."
        if not target.is_file():
            return f"Error: '{rel}' is not a file."
        return target.read_text(encoding="utf-8")

    async def delete(args: dict) -> str:
        rel = args.get("path")
        if not rel:
            return "Error: 'path' is required."
        try:
            target = _resolve_under_vault(vault, rel)
        except VaultError as e:
            return f"Error: {e}"
        if not target.exists():
            return f"Note '{rel}' already absent."
        if not target.is_file():
            return f"Error: '{rel}' is not a file."
        target.unlink()
        return f"Deleted: {rel}"

    return save, list_notes, search, read, delete, default_folder


def _read_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _make_snippet(text: str, query: str, width: int = 80) -> str:
    lc = text.lower()
    idx = lc.find(query.lower())
    if idx < 0:
        return text[:width].replace("\n", " ")
    start = max(0, idx - width // 2)
    end = min(len(text), idx + len(query) + width // 2)
    return text[start:end].replace("\n", " ")


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def make_obsidian_tools(
    vault: str | Path, default_folder: str = "Inbox"
) -> list[Tool]:
    """Build the five obsidian_* tools bound to a vault path.

    `default_folder` is where new notes land when no folder is specified —
    defaults to "Inbox" (PARA capture folder). Pass "" to revert to
    saving at the vault root.

    Raises VaultError if the vault path doesn't exist.
    """
    vault_path = Path(vault).expanduser().resolve()
    if not vault_path.is_dir():
        raise VaultError(f"vault '{vault}' is not a directory")

    save, list_notes, search, read, delete, default_folder = _make_handlers(
        vault_path, default_folder
    )

    return [
        Tool(
            definition=ToolDef(
                name="obsidian_save",
                description=(
                    f"Save markdown content to the Obsidian vault. Writes "
                    f"<vault>/<folder>/<title>.md with YAML frontmatter "
                    f"(title, created, tags, source). Notes default to the "
                    f"'{default_folder}' folder (PARA capture) — pass "
                    f"'folder' to file directly into Projects, Areas, "
                    f"Resources, Archive, or a subpath thereof when the "
                    f"target is clear."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Note title (used as filename)"},
                        "content": {"type": "string", "description": "Full markdown body — do not truncate"},
                        "folder": {"type": "string", "description": "Subfolder under vault root"},
                        "tags": {"type": "string", "description": "Comma-separated tags"},
                        "source": {"type": "string", "description": "Source URL if applicable"},
                    },
                    "required": ["title", "content"],
                },
            ),
            handler=save,
            kind="write",
        ),
        Tool(
            definition=ToolDef(
                name="obsidian_list",
                description=(
                    "List notes in the Obsidian vault. Returns "
                    "vault-relative paths with titles and tags. Call this "
                    "before claiming anything about what's in the vault."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "folder": {"type": "string", "description": "Subfolder to list (optional)"},
                    },
                    "required": [],
                },
            ),
            handler=list_notes,
            kind="read",
        ),
        Tool(
            definition=ToolDef(
                name="obsidian_search",
                description=(
                    "Full-text search across the Obsidian vault "
                    "(filenames and bodies). Returns vault-relative paths "
                    "with snippets. Use this to ground claims about "
                    "existing note content."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search term"},
                        "folder": {"type": "string", "description": "Optional subfolder"},
                    },
                    "required": ["query"],
                },
            ),
            handler=search,
            kind="read",
        ),
        Tool(
            definition=ToolDef(
                name="obsidian_read",
                description="Read the full content of a note by its vault-relative path.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Vault-relative path"},
                    },
                    "required": ["path"],
                },
            ),
            handler=read,
            kind="read",
        ),
        Tool(
            definition=ToolDef(
                name="obsidian_delete",
                description="Delete a note from the vault by its vault-relative path.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Vault-relative path"},
                    },
                    "required": ["path"],
                },
            ),
            handler=delete,
            kind="write",
        ),
    ]


# ---------------------------------------------------------------------------
# Skill bootstrap — called by ambi/cli/build.py at agent startup.
# ---------------------------------------------------------------------------


def register(tools: ToolRegistry) -> None:
    """Wire obsidian_* tools into the registry if OBSIDIAN_VAULT is set.

    Reads OBSIDIAN_VAULT (required) and OBSIDIAN_DEFAULT_FOLDER (optional,
    defaults to "Inbox"). Silently no-ops if no vault is configured; logs a
    warning to stderr if the path is invalid.
    """
    import os
    import sys

    vault = os.getenv("OBSIDIAN_VAULT")
    if not vault:
        return
    default_folder = os.getenv("OBSIDIAN_DEFAULT_FOLDER", "Inbox")
    try:
        for t in make_obsidian_tools(vault, default_folder=default_folder):
            tools.register(t)
    except VaultError as e:
        print(f"warning: obsidian tools not registered ({e})", file=sys.stderr)
