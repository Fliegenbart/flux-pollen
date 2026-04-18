"""ePIN Bayern ingestion.

The LGL Bayern operates a fully open, authenticated-free REST API for
the ePIN network (8 automated PomoAI pollen monitors + 4 Hirst traps).
Measurements are published in 3-hour buckets with units of pollen grains
per m³ — the continuous quantity we want to forecast, not the quantized
DWD index. The API accepts historical queries going back to at least
2019, which gives us a multi-year training window without a commercial
licence.

API docs (discovered from the client bundle):
    GET /api/locations          → station metadata (id, name, lat, lon)
    GET /api/pollen             → pollen species vocabulary
    GET /api/measurements?from=<unix>&to=<unix> → time series
    GET /api/seasons            → climatological seasons
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.metrics import ingestion_errors_total, ingestion_records_total
from app.core.time import utc_now
from app.models.database import PollenObservation
from app.services.data_ingest.region_mapping import (
    EPIN_POLLEN_SCIENTIFIC_TO_CANONICAL,
    EPIN_STATION_REGION,
    EPIN_STATIONS,
)

logger = logging.getLogger(__name__)

EPIN_API_BASE = "https://epin.lgl.bayern.de/api"
SOURCE_NETWORK = "ePIN"


class EPINIngestError(RuntimeError):
    """Raised when the ePIN payload is unreachable or malformed beyond recovery."""


class EPINService:
    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #

    def run_full_import(
        self,
        *,
        from_time: datetime,
        to_time: datetime | None = None,
        api_base: str | None = None,
    ) -> dict[str, Any]:
        """Fetch measurements over [from_time, to_time] and upsert."""
        to_time = to_time or utc_now()
        base = (api_base or EPIN_API_BASE).rstrip("/")
        try:
            payload = self._fetch_measurements(base, from_time, to_time)
        except Exception as exc:
            ingestion_errors_total.labels(source="epin").inc()
            logger.error("ePIN fetch failed [%s .. %s]: %s", from_time, to_time, exc)
            return {
                "success": False,
                "error": f"ePIN API unreachable: {exc}",
                "from_time": from_time.isoformat(),
                "to_time": to_time.isoformat(),
                "timestamp": utc_now().isoformat(),
            }
        return self._ingest_payload(payload, source_url=f"{base}/measurements")

    def import_from_file(self, path: str | Path) -> dict[str, Any]:
        """Ingest a captured ePIN payload (fixtures, cold-backfill snapshots)."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return self._ingest_payload(payload, source_url=f"file://{path}")

    def backfill_range(
        self,
        *,
        start: datetime,
        end: datetime,
        chunk_days: int = 30,
        api_base: str | None = None,
    ) -> dict[str, Any]:
        """Walk a long date range in chunks to stay under any per-request cap."""
        base = (api_base or EPIN_API_BASE).rstrip("/")
        totals = {"success": True, "inserted": 0, "updated": 0, "chunks": 0, "errors": []}
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=max(int(chunk_days), 1)), end)
            try:
                payload = self._fetch_measurements(base, cursor, chunk_end)
                result = self._ingest_payload(payload, source_url=f"{base}/measurements")
                totals["inserted"] += int(result.get("inserted", 0))
                totals["updated"] += int(result.get("updated", 0))
                totals["chunks"] += 1
            except Exception as exc:
                ingestion_errors_total.labels(source="epin").inc()
                totals["success"] = False
                totals["errors"].append(f"{cursor.date()}..{chunk_end.date()}: {exc}")
            cursor = chunk_end
        totals["timestamp"] = utc_now().isoformat()
        return totals

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, requests.HTTPError)
        ),
    )
    def _fetch_measurements(self, base: str, from_time: datetime, to_time: datetime) -> dict[str, Any]:
        params = {
            "from": int(_to_epoch(from_time)),
            "to": int(_to_epoch(to_time)),
        }
        url = f"{base}/measurements"
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ #
    # Payload → rows
    # ------------------------------------------------------------------ #

    def _ingest_payload(self, payload: dict[str, Any], *, source_url: str) -> dict[str, Any]:
        ingest_time = utc_now()
        series = payload.get("measurements") or []
        if not isinstance(series, list):
            raise EPINIngestError(f"Unexpected 'measurements' type: {type(series).__name__}")

        rows = list(self._series_to_rows(series, ingest_time=ingest_time))
        inserted, updated = self._upsert(rows)
        ingestion_records_total.labels(source="epin").inc(inserted + updated)

        return {
            "success": inserted + updated > 0,
            "source_url": source_url,
            "series_count": len(series),
            "records_total": len(rows),
            "inserted": inserted,
            "updated": updated,
            "stations": sorted({row["station_id"] for row in rows}),
            "pollen_types": sorted({row["pollen_type"] for row in rows}),
            "window": {
                "from": payload.get("from"),
                "to": payload.get("to"),
            },
            "ingest_time": ingest_time.isoformat(),
        }

    def _series_to_rows(
        self,
        series: Iterable[dict[str, Any]],
        *,
        ingest_time: datetime,
    ) -> Iterable[dict[str, Any]]:
        for entry in series:
            if not isinstance(entry, dict):
                continue
            station_id = str(entry.get("location") or "").strip().upper()
            if station_id not in EPIN_STATIONS:
                # Unknown station — log once and skip; the API may grow.
                logger.debug("Skipping unknown ePIN station %r", station_id)
                continue
            pollen_canonical = EPIN_POLLEN_SCIENTIFIC_TO_CANONICAL.get(
                str(entry.get("polle") or "").strip()
            )
            if not pollen_canonical:
                continue

            station_name = EPIN_STATIONS[station_id]
            region_code = EPIN_STATION_REGION[station_id]
            data_points = entry.get("data") or []
            if not isinstance(data_points, list):
                continue

            for point in data_points:
                if not isinstance(point, dict):
                    continue
                value = point.get("value")
                if value is None:
                    continue
                from_epoch = point.get("from")
                to_epoch = point.get("to")
                if from_epoch is None or to_epoch is None:
                    continue
                try:
                    from_time = _from_epoch(int(from_epoch))
                    to_time = _from_epoch(int(to_epoch))
                except (TypeError, ValueError):
                    continue

                yield {
                    "station_id": station_id,
                    "station_name": station_name,
                    "region_code": region_code,
                    "pollen_type": pollen_canonical,
                    "from_time": from_time,
                    "to_time": to_time,
                    "concentration": float(value),
                    "algorithm": str(point.get("algorithm") or "").strip() or None,
                    "source_network": SOURCE_NETWORK,
                    "available_time": ingest_time,
                }

    # ------------------------------------------------------------------ #
    # Upsert (dialect-aware, mirrors PollenService)
    # ------------------------------------------------------------------ #

    def _upsert(self, rows: list[dict[str, Any]]) -> tuple[int, int]:
        if not rows:
            return 0, 0
        dialect = self.db.bind.dialect.name if self.db.bind is not None else ""
        if dialect == "postgresql":
            return self._upsert_postgres(rows)
        return self._upsert_generic(rows)

    def _upsert_postgres(self, rows: list[dict[str, Any]]) -> tuple[int, int]:
        keys = {(r["station_id"], r["pollen_type"], r["from_time"]) for r in rows}
        existing = (
            self.db.query(
                PollenObservation.station_id,
                PollenObservation.pollen_type,
                PollenObservation.from_time,
            )
            .filter(
                PollenObservation.station_id.in_({k[0] for k in keys}),
                PollenObservation.pollen_type.in_({k[1] for k in keys}),
                PollenObservation.from_time.in_({k[2] for k in keys}),
            )
            .all()
        )
        existing_count = len({tuple(row) for row in existing}.intersection(keys))

        stmt = pg_insert(PollenObservation).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_pollen_obs_station_type_window",
            set_={
                "concentration": stmt.excluded.concentration,
                "algorithm": stmt.excluded.algorithm,
                "available_time": stmt.excluded.available_time,
                "to_time": stmt.excluded.to_time,
                "station_name": stmt.excluded.station_name,
                "region_code": stmt.excluded.region_code,
            },
        )
        self.db.execute(stmt)
        self.db.commit()
        return len(rows) - existing_count, existing_count

    def _upsert_generic(self, rows: list[dict[str, Any]]) -> tuple[int, int]:
        inserted = 0
        updated = 0
        for record in rows:
            existing = (
                self.db.query(PollenObservation)
                .filter(
                    PollenObservation.station_id == record["station_id"],
                    PollenObservation.pollen_type == record["pollen_type"],
                    PollenObservation.from_time == record["from_time"],
                )
                .one_or_none()
            )
            if existing is not None:
                existing.concentration = record["concentration"]
                existing.algorithm = record["algorithm"]
                existing.available_time = record["available_time"]
                existing.to_time = record["to_time"]
                existing.station_name = record["station_name"]
                existing.region_code = record["region_code"]
                updated += 1
            else:
                self.db.add(PollenObservation(**record))
                inserted += 1
        self.db.commit()
        return inserted, updated


def _to_epoch(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _from_epoch(seconds: int) -> datetime:
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc).replace(tzinfo=None)
