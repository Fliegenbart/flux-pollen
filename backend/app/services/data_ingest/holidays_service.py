"""School-holiday ingestion via schulferien-api.de (v2).

Holidays drive strong behavioral seasonality on OTC search and
purchasing, and they shift the effective exposure population for a
pollen wave (kids and families move differently in holidays). The service
keeps the 16 Bundesländer covered for the relevant training horizon.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import requests
from sqlalchemy import func
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.metrics import ingestion_errors_total, ingestion_records_total
from app.core.time import utc_now
from app.models.database import SchoolHolidays
from app.services.data_ingest.region_mapping import ALL_BUNDESLAENDER

logger = logging.getLogger(__name__)

API_BASE = "https://schulferien-api.de/api/v2"


class SchoolHolidaysService:
    def __init__(self, db: Session):
        self.db = db

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, requests.HTTPError)
        ),
    )
    def fetch_year(self, year: int) -> list[dict[str, Any]]:
        url = f"{API_BASE}/{year}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def import_year(self, year: int) -> int:
        entries = self.fetch_year(year)
        if not entries:
            return 0

        count_new = 0
        for entry in entries:
            state_code = str(entry.get("stateCode") or "").strip()
            if state_code not in ALL_BUNDESLAENDER:
                continue
            ferien_typ = entry.get("name_cp") or entry.get("name") or "Sonstige"
            start_str = str(entry.get("start") or "")[:10]
            end_str = str(entry.get("end") or "")[:10]

            try:
                start_datum = datetime.fromisoformat(start_str)
                end_datum = datetime.fromisoformat(end_str)
            except ValueError:
                logger.warning("Skipping invalid date in holidays payload: %r..%r", start_str, end_str)
                continue

            existing = (
                self.db.query(SchoolHolidays)
                .filter(
                    SchoolHolidays.bundesland == state_code,
                    SchoolHolidays.ferien_typ == ferien_typ,
                    SchoolHolidays.jahr == year,
                    SchoolHolidays.start_datum == start_datum,
                )
                .one_or_none()
            )

            if existing:
                if existing.end_datum != end_datum:
                    existing.end_datum = end_datum
                continue

            self.db.add(
                SchoolHolidays(
                    bundesland=state_code,
                    ferien_typ=ferien_typ,
                    start_datum=start_datum,
                    end_datum=end_datum,
                    jahr=year,
                )
            )
            count_new += 1

        self.db.commit()
        return count_new

    def run_full_import(self, years: list[int] | None = None) -> dict[str, Any]:
        if years is None:
            current_year = datetime.now().year
            years = [current_year - 1, current_year, current_year + 1]

        total_new = 0
        errors: list[str] = []
        for year in years:
            try:
                total_new += self.import_year(year)
            except Exception as exc:
                ingestion_errors_total.labels(source="holidays").inc()
                logger.error("Holidays import failed for %s: %s", year, exc)
                self.db.rollback()
                errors.append(f"{year}: {exc}")

        per_state = dict(
            self.db.query(SchoolHolidays.bundesland, func.count(SchoolHolidays.id))
            .group_by(SchoolHolidays.bundesland)
            .all()
        )
        ingestion_records_total.labels(source="holidays").inc(total_new)
        return {
            "success": not errors,
            "years": years,
            "new_entries": total_new,
            "total_in_db": self.db.query(SchoolHolidays).count(),
            "states_covered": len(per_state),
            "per_state": per_state,
            "errors": errors or None,
            "timestamp": utc_now().isoformat(),
        }

    def is_holiday(self, target: date, bundesland: str | None = None) -> bool:
        query = self.db.query(SchoolHolidays).filter(
            SchoolHolidays.start_datum <= target,
            SchoolHolidays.end_datum >= target,
        )
        if bundesland:
            query = query.filter(SchoolHolidays.bundesland == bundesland)
        return query.first() is not None

    def upcoming_school_starts(self, days_ahead: int = 30) -> list[dict[str, Any]]:
        now = datetime.now()
        window_end = now + timedelta(days=days_ahead)
        entries = (
            self.db.query(SchoolHolidays)
            .filter(SchoolHolidays.end_datum >= now, SchoolHolidays.end_datum <= window_end)
            .order_by(SchoolHolidays.end_datum.asc())
            .all()
        )
        return [
            {
                "bundesland": entry.bundesland,
                "ferien_typ": entry.ferien_typ,
                "end_datum": entry.end_datum.isoformat(),
                "school_start": (entry.end_datum + timedelta(days=1)).strftime("%Y-%m-%d"),
                "days_until": (entry.end_datum - now).days,
            }
            for entry in entries
        ]
