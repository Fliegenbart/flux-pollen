"""DWD pollen ingestion.

Pulls the daily alert file https://opendata.dwd.de/.../s31fg.json,
normalizes it onto the 16-Bundesländer axis, and upserts into
``pollen_data`` with ``available_time`` stamped at ingest time so we keep
the Point-in-Time semantics the forecast pipeline depends on.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.metrics import ingestion_errors_total, ingestion_records_total
from app.core.time import utc_now
from app.models.database import PollenData
from app.services.data_ingest.region_mapping import dwd_region_to_codes

logger = logging.getLogger(__name__)

# DWD publishes today (offset 0), tomorrow (+1) and dayafter (+2).
_DAY_OFFSETS: dict[str, int] = {
    "today": 0,
    "tomorrow": 1,
    "dayafter_to": 2,
}

_EMPTY_INDEX_TOKENS = {"", "-", "keine", "k.a.", "na", "n/a"}

SUPPORTED_POLLEN_TYPES: tuple[str, ...] = (
    "hasel",
    "erle",
    "esche",
    "birke",
    "graeser",
    "roggen",
    "beifuss",
    "ambrosia",
)

_POLLEN_TYPE_ALIASES: dict[str, str] = {
    "hasel": "hasel",
    "erle": "erle",
    "esche": "esche",
    "birke": "birke",
    "graeser": "graeser",
    "gräser": "graeser",
    "roggen": "roggen",
    "beifuss": "beifuss",
    "beifuß": "beifuss",
    "ambrosia": "ambrosia",
}


class PollenIngestError(RuntimeError):
    """Raised when the DWD payload is unreachable or malformed beyond recovery."""


class PollenService:
    """DWD pollen OpenData ingester.

    Hardened vs. the original stub: tenacity-based retries, dialect-aware
    upserts, strict region/pollen-type normalization, and a payload-from-file
    path for deterministic testing.
    """

    def __init__(self, db: Session):
        self.db = db
        self._settings = get_settings()

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #

    def run_full_import(self, *, source_url: str | None = None) -> dict[str, Any]:
        """Fetch the current DWD payload and persist it. Idempotent."""
        url = source_url or self._settings.DWD_POLLEN_URL
        try:
            payload = self._fetch_payload(url)
        except Exception as exc:
            ingestion_errors_total.labels(source="pollen").inc()
            logger.error("Pollen fetch failed (%s): %s", url, exc)
            return {
                "success": False,
                "error": f"Pollen source unreachable: {exc}",
                "source_url": url,
                "timestamp": utc_now().isoformat(),
            }
        return self._ingest_payload(payload, source_url=url)

    def import_from_file(self, path: str | Path) -> dict[str, Any]:
        """Ingest a DWD payload stored locally (fixtures, archived snapshots)."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return self._ingest_payload(payload, source_url=f"file://{path}")

    # ------------------------------------------------------------------ #
    # HTTP fetch (retry-wrapped)
    # ------------------------------------------------------------------ #

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, requests.HTTPError)
        ),
    )
    def _fetch_payload(self, url: str) -> dict[str, Any]:
        logger.info("Fetching DWD pollen payload from %s", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ #
    # Core ingestion
    # ------------------------------------------------------------------ #

    def _ingest_payload(self, payload: dict[str, Any], *, source_url: str) -> dict[str, Any]:
        ingest_time = utc_now()
        last_update = self._parse_dwd_timestamp(payload.get("last_update"))
        base_date = (last_update or ingest_time).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        records = self._parse_records(
            payload.get("content") or [],
            base_date=base_date,
            ingest_time=ingest_time,
        )

        inserted, updated = self._upsert(records)
        ingestion_records_total.labels(source="pollen").inc(inserted + updated)

        latest_target_date = (
            max((row["datum"] for row in records), default=None) if records else None
        )
        return {
            "success": inserted + updated > 0,
            "source_url": source_url,
            "records_total": len(records),
            "inserted": inserted,
            "updated": updated,
            "regions": sorted({row["region_code"] for row in records}),
            "pollen_types": sorted({row["pollen_type"] for row in records}),
            "last_update": last_update.isoformat() if last_update else None,
            "latest_target_date": latest_target_date.isoformat() if latest_target_date else None,
            "ingest_time": ingest_time.isoformat(),
        }

    def _parse_records(
        self,
        content: list[dict[str, Any]],
        *,
        base_date: datetime,
        ingest_time: datetime,
    ) -> list[dict[str, Any]]:
        seen: dict[tuple[str, str, datetime], dict[str, Any]] = {}
        for region_entry in content:
            region_codes = dwd_region_to_codes(region_entry.get("region_name"))
            if not region_codes:
                continue

            pollen_block = region_entry.get("Pollen") or {}
            if not isinstance(pollen_block, dict):
                continue

            for raw_pollen, horizon_map in pollen_block.items():
                pollen_type = self._normalize_pollen_type(raw_pollen)
                if not pollen_type or not isinstance(horizon_map, dict):
                    continue

                for day_key, offset in _DAY_OFFSETS.items():
                    index_value = self._parse_index(horizon_map.get(day_key))
                    if index_value is None:
                        continue
                    datum = base_date + timedelta(days=offset)
                    for code in region_codes:
                        key = (code, pollen_type, datum)
                        # If the same (region, pollen, date) appears twice (e.g.
                        # because two source labels map to the same state),
                        # keep the most permissive (max) index — DWD-aligned.
                        existing_record = seen.get(key)
                        if existing_record is None or index_value > existing_record["pollen_index"]:
                            seen[key] = {
                                "region_code": code,
                                "pollen_type": pollen_type,
                                "datum": datum,
                                "available_time": ingest_time,
                                "pollen_index": index_value,
                                "source": "DWD",
                            }
        return list(seen.values())

    # ------------------------------------------------------------------ #
    # Upsert — dialect-aware (PostgreSQL fast path, SQLite fallback for tests)
    # ------------------------------------------------------------------ #

    def _upsert(self, rows: list[dict[str, Any]]) -> tuple[int, int]:
        if not rows:
            return 0, 0

        dialect = self.db.bind.dialect.name if self.db.bind is not None else ""
        if dialect == "postgresql":
            return self._upsert_postgres(rows)
        return self._upsert_generic(rows)

    def _upsert_postgres(self, rows: list[dict[str, Any]]) -> tuple[int, int]:
        """Single-statement upsert on PostgreSQL using ``ON CONFLICT``."""
        stmt = pg_insert(PollenData).values(rows)
        update_cols = {
            "pollen_index": stmt.excluded.pollen_index,
            "available_time": stmt.excluded.available_time,
            "source": stmt.excluded.source,
        }
        stmt = stmt.on_conflict_do_update(
            constraint="uq_pollen_region_type_date",
            set_=update_cols,
        )

        # We want to report insert-vs-update counts. Postgres 15+ lets us
        # ask via RETURNING xmax — but SQLAlchemy-compat is fiddly, so we
        # pre-count existing keys instead.
        keys = [(r["region_code"], r["pollen_type"], r["datum"]) for r in rows]
        existing = self._count_existing(keys)
        self.db.execute(stmt)
        self.db.commit()

        inserted = len(rows) - existing
        return inserted, existing

    def _upsert_generic(self, rows: list[dict[str, Any]]) -> tuple[int, int]:
        inserted = 0
        updated = 0
        for record in rows:
            existing = (
                self.db.query(PollenData)
                .filter(
                    PollenData.region_code == record["region_code"],
                    PollenData.pollen_type == record["pollen_type"],
                    PollenData.datum == record["datum"],
                )
                .one_or_none()
            )
            if existing:
                existing.pollen_index = record["pollen_index"]
                existing.available_time = record["available_time"]
                existing.source = record["source"]
                updated += 1
            else:
                self.db.add(PollenData(**record))
                inserted += 1
        self.db.commit()
        return inserted, updated

    def _count_existing(self, keys: list[tuple[str, str, datetime]]) -> int:
        if not keys:
            return 0
        query = select(
            PollenData.region_code, PollenData.pollen_type, PollenData.datum
        ).where(
            PollenData.region_code.in_({k[0] for k in keys}),
            PollenData.pollen_type.in_({k[1] for k in keys}),
            PollenData.datum.in_({k[2] for k in keys}),
        )
        existing = {tuple(row) for row in self.db.execute(query).all()}
        return len(existing.intersection(keys))

    # ------------------------------------------------------------------ #
    # Parsers (static helpers)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_dwd_timestamp(raw: Any) -> datetime | None:
        if not raw:
            return None
        text = str(raw).replace(" Uhr", "").strip()
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_index(raw: Any) -> float | None:
        """Parse DWD index tokens such as "0", "1-2", "2" into a float.

        Ranges ("0-1", "1-2") are averaged. The DWD emits a handful of
        empty markers we explicitly ignore.
        """
        if raw is None:
            return None
        text = str(raw).strip().lower()
        if text in _EMPTY_INDEX_TOKENS:
            return None
        matches = re.findall(r"\d+(?:[.,]\d+)?", text)
        if not matches:
            return None
        values = [float(m.replace(",", ".")) for m in matches]
        return round(sum(values) / len(values), 3)

    @staticmethod
    def _normalize_pollen_type(raw: Any) -> str | None:
        token = str(raw or "").strip().lower()
        if not token:
            return None
        return _POLLEN_TYPE_ALIASES.get(token)
