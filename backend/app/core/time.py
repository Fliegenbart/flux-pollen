from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return a naive UTC datetime with the same storage semantics used throughout the app."""
    return datetime.now(UTC).replace(tzinfo=None)


def utc_now_iso() -> str:
    return utc_now().isoformat()
