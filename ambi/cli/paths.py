"""Resolve where ambi keeps its per-user state.

Default: ``~/.ambi/``. Override the whole tree with ``AMBI_HOME``.

Layout::

    ~/.ambi/
        .env              # user secrets (loaded by load_env at startup)
        skills/           # user-authored skills (markdown)
        data/
            session.db    # SqliteStore session
            tasks.db      # Scheduler TaskStore
            hippocamp.log # Hippocamp subprocess stderr
"""

from __future__ import annotations

import os
from pathlib import Path


def ambi_home() -> Path:
    raw = os.getenv("AMBI_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".ambi"


def env_file() -> Path:
    return ambi_home() / ".env"


def system_md() -> Path:
    """User-editable system prompt override. Falls back to default if missing."""
    return ambi_home() / "system.md"


def skills_dir() -> Path:
    return ambi_home() / "skills"


def data_dir() -> Path:
    return ambi_home() / "data"


def session_db() -> Path:
    return data_dir() / "session.db"


def tasks_db() -> Path:
    return data_dir() / "tasks.db"


def hippocamp_log() -> Path:
    return data_dir() / "hippocamp.log"


def ensure_tree() -> None:
    """Make ambi_home, skills/, and data/ if missing. Idempotent."""
    for d in (ambi_home(), skills_dir(), data_dir()):
        d.mkdir(parents=True, exist_ok=True)
