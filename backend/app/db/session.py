"""Database session + engine. Schema is managed exclusively by Alembic."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _build_engine(database_url: str):
    """Build an engine sized appropriately for the backend dialect.

    SQLite has no concept of a server-side connection pool and reacts
    poorly to QueuePool under concurrent workers — we use a thread-safe
    in-process pool instead. Postgres keeps the sturdier QueuePool with
    pool_pre_ping for liveness checks across long-lived connections.
    """
    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
    return create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=settings.ENVIRONMENT == "development",
    )


engine = _build_engine(settings.DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def check_db_connection() -> bool:
    try:
        with get_db_context() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database connection check failed: %s", exc)
        return False
