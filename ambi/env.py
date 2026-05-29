"""Optional .env loader.

Apps call `load_env()` once at startup to populate `os.environ` from a `.env`
file. Library code never auto-loads — that's the caller's choice.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env(path: str | Path | None = None, override: bool = False) -> None:
    """Load `.env` into `os.environ`.

    If `path` is None, walks up from the cwd looking for a `.env` file
    (python-dotenv default behaviour).
    """
    if path is None:
        load_dotenv(override=override)
    else:
        load_dotenv(dotenv_path=path, override=override)


def require_env(key: str) -> str:
    """Return env var or raise — for fail-fast at startup."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Required environment variable {key!r} is not set.")
    return value
