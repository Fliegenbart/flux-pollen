"""Test harness: put the backend root on sys.path and seed env vars.

All data tests use an in-memory SQLite instance. Production runs on
PostgreSQL; the SQLite path is only there to keep unit tests fast and
hermetic. The upsert code in PollenService branches on dialect so it
exercises the right path either way.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Settings that the config module insists on resolving before any code runs.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("SECRET_KEY", "test-secret-key-minimum-32-characters-long")
os.environ.setdefault("ADMIN_EMAIL", "admin@example.com")
os.environ.setdefault("ADMIN_PASSWORD", "test-password-12345")
os.environ.setdefault("ENVIRONMENT", "test")
# Keep the module-level engine in app.db.session on a dialect that never
# needs a driver we may not have installed in CI. The per-test fixture
# below creates its own engine and dependency-overrides it for API calls.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.database import Base


@pytest.fixture
def db_session():
    # StaticPool + check_same_thread=False lets the FastAPI TestClient's
    # worker thread reuse the same in-memory DB that the test thread created.
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def dwd_pollen_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "dwd_pollen_sample.json"
