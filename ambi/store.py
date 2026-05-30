"""SqliteStore — durable session history.

Schema is a single `messages` table keyed by `(session_id, seq)`. Each row
stores one Message with its content (a list of Blocks) JSON-encoded.

Block discriminator on the type field:
    text         → TextBlock
    tool_use     → ToolUseBlock
    tool_result  → ToolResultBlock
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from .types import (
    Block,
    CompactionAnchor,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


async def enable_wal(db) -> None:
    """Best-effort WAL journal mode + a busy timeout on an open connection.

    WAL lets readers and a single writer proceed concurrently — fewer
    "database is locked" errors when chat, scheduler, and usage tracking touch
    SQLite at once. But switching *into* WAL needs a brief exclusive lock and
    returns SQLITE_BUSY immediately if another connection has the file open
    (common at daemon startup). So this is best-effort and never fatal:
    rollback-journal mode is fully correct, just less concurrent, and the
    busy_timeout gives the transition a chance to wait out short-lived
    connections. WAL, once set, persists on the file.
    """
    try:
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL DEFAULT 'default',
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);

CREATE TABLE IF NOT EXISTS compaction_anchors (
    session_id TEXT NOT NULL DEFAULT 'default',
    from_seq   INTEGER NOT NULL,
    to_seq     INTEGER NOT NULL,
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, from_seq, to_seq)
);
CREATE INDEX IF NOT EXISTS idx_anchors_session ON compaction_anchors(session_id, from_seq);
"""


class SqliteStore:
    """Async SQLite-backed session store."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._initialized = False

    async def _ensure(self) -> None:
        if self._initialized:
            return
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await enable_wal(db)
            await db.executescript(_SCHEMA)
            await db.commit()
        self._initialized = True

    async def load(self, session_id: str = "default") -> list[Message]:
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? ORDER BY seq ASC",
                (session_id,),
            )
            rows = await cur.fetchall()
        return [
            Message(role=role, content=_decode_content(content))
            for role, content in rows
        ]

    async def append(
        self, messages: list[Message], session_id: str = "default"
    ) -> None:
        """Append messages after the current max seq for this session."""
        if not messages:
            return
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE session_id = ?",
                (session_id,),
            )
            (max_seq,) = await cur.fetchone()
            await db.executemany(
                "INSERT INTO messages (session_id, seq, role, content) "
                "VALUES (?, ?, ?, ?)",
                [
                    (session_id, max_seq + 1 + i, msg.role, _encode_content(msg.content))
                    for i, msg in enumerate(messages)
                ],
            )
            await db.commit()

    async def clear(self, session_id: str = "default") -> None:
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,),
            )
            await db.execute(
                "DELETE FROM compaction_anchors WHERE session_id = ?",
                (session_id,),
            )
            await db.commit()

    # ----- compaction anchors -----

    async def save_anchor(
        self, anchor: CompactionAnchor, session_id: str = "default"
    ) -> None:
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO compaction_anchors "
                "(session_id, from_seq, to_seq, summary) VALUES (?, ?, ?, ?)",
                (session_id, anchor.from_seq, anchor.to_seq, anchor.summary),
            )
            await db.commit()

    async def load_anchors(
        self, session_id: str = "default"
    ) -> list[CompactionAnchor]:
        await self._ensure()
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                "SELECT from_seq, to_seq, summary, created_at "
                "FROM compaction_anchors WHERE session_id = ? "
                "ORDER BY from_seq ASC",
                (session_id,),
            )
            rows = await cur.fetchall()
        return [
            CompactionAnchor(
                from_seq=r[0], to_seq=r[1], summary=r[2], created_at=r[3] or "",
            )
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Block <-> JSON serialization
# ---------------------------------------------------------------------------


def _encode_content(blocks: list[Block]) -> str:
    return json.dumps([_block_to_dict(b) for b in blocks])


def _decode_content(raw: str) -> list[Block]:
    return [_block_from_dict(d) for d in json.loads(raw)]


def _block_to_dict(block: Block) -> dict:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
            "tool_name": block._tool_name,
        }
    raise TypeError(f"Cannot serialize block: {type(block).__name__}")


def _block_from_dict(data: dict) -> Block:
    kind = data["type"]
    if kind == "text":
        return TextBlock(text=data["text"])
    if kind == "tool_use":
        return ToolUseBlock(
            id=data["id"], name=data["name"], input=data["input"],
        )
    if kind == "tool_result":
        return ToolResultBlock(
            tool_use_id=data["tool_use_id"],
            content=data["content"],
            is_error=data.get("is_error", False),
            _tool_name=data.get("tool_name", ""),
        )
    raise ValueError(f"Unknown block type: {kind!r}")
