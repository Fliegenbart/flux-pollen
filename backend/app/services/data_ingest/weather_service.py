"""Weather ingestion via BrightSky (free DWD mirror, no API key).

Covers the 16 Landeshauptstädte; each observation is stamped with
``region_code`` so downstream feature builders can join pollen and
weather by Bundesland without a separate lookup.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import requests
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.metrics import ingestion_errors_total, ingestion_records_total
from app.core.time import utc_now
from app.models.database import WeatherData
from app.services.data_ingest.region_mapping import CAPITAL_TO_CODE

logger = logging.getLogger(__name__)

BRIGHTSKY_BASE = "https://api.brightsky.dev"
FORECAST_HORIZON_DAYS = 8
CURRENT_DATA_TYPE = "CURRENT"
DAILY_OBSERVATION_DATA_TYPE = "DAILY_OBSERVATION"
DAILY_FORECAST_DATA_TYPE = "DAILY_FORECAST"

# Each capital city coordinate serves as the regional proxy.
CITIES: list[dict[str, Any]] = [
    {"name": "Kiel", "lat": 54.32, "lon": 10.14},
    {"name": "Hamburg", "lat": 53.55, "lon": 9.99},
    {"name": "Schwerin", "lat": 53.63, "lon": 11.41},
    {"name": "Bremen", "lat": 53.08, "lon": 8.80},
    {"name": "Hannover", "lat": 52.37, "lon": 9.74},
    {"name": "Berlin", "lat": 52.52, "lon": 13.41},
    {"name": "Potsdam", "lat": 52.40, "lon": 13.07},
    {"name": "Magdeburg", "lat": 52.13, "lon": 11.63},
    {"name": "Dresden", "lat": 51.05, "lon": 13.74},
    {"name": "Erfurt", "lat": 50.98, "lon": 11.03},
    {"name": "Düsseldorf", "lat": 51.23, "lon": 6.78},
    {"name": "Saarbrücken", "lat": 49.23, "lon": 7.00},
    {"name": "Wiesbaden", "lat": 50.08, "lon": 8.24},
    {"name": "Mainz", "lat": 50.00, "lon": 8.27},
    {"name": "Stuttgart", "lat": 48.78, "lon": 9.18},
    {"name": "München", "lat": 48.14, "lon": 11.58},
]


class WeatherService:
    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------ #
    # HTTP (retry-wrapped)
    # ------------------------------------------------------------------ #

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, requests.HTTPError)
        ),
    )
    def _fetch_range(self, city: dict[str, Any], date_str: str, last_date_str: str) -> list[dict]:
        url = f"{BRIGHTSKY_BASE}/weather"
        params = {
            "lat": city["lat"],
            "lon": city["lon"],
            "date": date_str,
            "last_date": last_date_str,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("weather", [])

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, requests.HTTPError)
        ),
    )
    def _fetch_current(self, city: dict[str, Any]) -> dict[str, Any] | None:
        url = f"{BRIGHTSKY_BASE}/current_weather"
        params = {"lat": city["lat"], "lon": city["lon"]}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("weather")

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mean(hourly: list[dict[str, Any]], key: str) -> float | None:
        vals = [r[key] for r in hourly if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    @staticmethod
    def _sum(hourly: list[dict[str, Any]], key: str) -> float | None:
        vals = [r[key] for r in hourly if r.get(key) is not None]
        return round(sum(vals), 2) if vals else None

    def _aggregate_day(
        self, hourly: list[dict[str, Any]], *, city: str, day: datetime
    ) -> dict[str, Any]:
        wind_kmh = self._mean(hourly, "wind_speed")
        return {
            "city": city,
            "region_code": CAPITAL_TO_CODE.get(city),
            "datum": day.replace(hour=12, minute=0, second=0, microsecond=0),
            "available_time": day.replace(hour=23, minute=59, second=0, microsecond=0),
            "temperatur": self._mean(hourly, "temperature"),
            "gefuehlte_temperatur": None,
            "luftfeuchtigkeit": self._mean(hourly, "relative_humidity"),
            "luftdruck": self._mean(hourly, "pressure_msl"),
            "wind_geschwindigkeit": round(wind_kmh / 3.6, 2) if wind_kmh is not None else None,
            "wolken": self._mean(hourly, "cloud_cover"),
            "niederschlag_wahrscheinlichkeit": None,
            "regen_mm": self._sum(hourly, "precipitation"),
            "taupunkt": self._mean(hourly, "dew_point"),
        }

    # ------------------------------------------------------------------ #
    # Upsert — shared by current/observation/forecast
    # ------------------------------------------------------------------ #

    def _upsert(self, record: dict[str, Any], *, data_type: str) -> str:
        query = self.db.query(WeatherData).filter(
            WeatherData.city == record["city"],
            WeatherData.datum == record["datum"],
            WeatherData.data_type == data_type,
        )

        is_forecast = data_type.endswith("FORECAST")
        forecast_run_id = record.get("forecast_run_id")
        forecast_run_ts = record.get("forecast_run_timestamp")
        if is_forecast and forecast_run_id:
            query = query.filter(WeatherData.forecast_run_id == forecast_run_id)
        elif is_forecast and forecast_run_ts is not None:
            query = query.filter(WeatherData.forecast_run_timestamp == forecast_run_ts)

        existing = query.one_or_none()
        if existing is not None:
            incoming_available = record.get("available_time")
            if incoming_available is not None and (
                existing.available_time is None or incoming_available < existing.available_time
            ):
                existing.available_time = incoming_available
            for key, value in record.items():
                if key in {"city", "datum", "available_time"} or value is None:
                    continue
                setattr(existing, key, value)
            return "updated"

        payload = dict(record)
        payload.setdefault("available_time", utc_now())
        payload["data_type"] = data_type
        self.db.add(WeatherData(**payload))
        return "inserted"

    # ------------------------------------------------------------------ #
    # Import entry points
    # ------------------------------------------------------------------ #

    def import_current(self) -> dict[str, int]:
        inserted = 0
        updated = 0
        for city in CITIES:
            try:
                current = self._fetch_current(city)
            except Exception as exc:
                ingestion_errors_total.labels(source="weather").inc()
                logger.warning("Current weather fetch failed for %s: %s", city["name"], exc)
                continue
            if not current:
                continue

            wind_kmh = current.get("wind_speed_60") or current.get("wind_speed_30")
            record = {
                "city": city["name"],
                "region_code": CAPITAL_TO_CODE.get(city["name"]),
                "datum": utc_now().replace(minute=0, second=0, microsecond=0),
                "available_time": utc_now(),
                "temperatur": current.get("temperature"),
                "luftfeuchtigkeit": current.get("relative_humidity"),
                "luftdruck": current.get("pressure_msl"),
                "wind_geschwindigkeit": round(wind_kmh / 3.6, 2) if wind_kmh else None,
                "wolken": current.get("cloud_cover"),
                "regen_mm": current.get("precipitation_60"),
                "taupunkt": current.get("dew_point"),
            }
            outcome = self._upsert(record, data_type=CURRENT_DATA_TYPE)
            if outcome == "inserted":
                inserted += 1
            else:
                updated += 1

        self.db.commit()
        ingestion_records_total.labels(source="weather").inc(inserted + updated)
        return {"inserted": inserted, "updated": updated}

    def import_forecast(self) -> dict[str, int]:
        tomorrow = (utc_now() + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = tomorrow + timedelta(days=FORECAST_HORIZON_DAYS)
        forecast_run_ts = utc_now().replace(second=0, microsecond=0)
        forecast_run_id = f"weather_forecast_run:{forecast_run_ts.isoformat()}"

        inserted = 0
        updated = 0
        for city in CITIES:
            try:
                records = self._fetch_range(
                    city,
                    tomorrow.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"),
                )
            except Exception as exc:
                ingestion_errors_total.labels(source="weather").inc()
                logger.warning("Forecast fetch failed for %s: %s", city["name"], exc)
                continue

            by_day: dict[str, list[dict]] = {}
            for row in records:
                day_key = str(row.get("timestamp") or "")[:10]
                if not day_key:
                    continue
                by_day.setdefault(day_key, []).append(row)

            for day_str, hourly in by_day.items():
                if len(hourly) < 6:
                    continue
                day = datetime.strptime(day_str, "%Y-%m-%d")
                aggregated = self._aggregate_day(hourly, city=city["name"], day=day)
                aggregated.update(
                    forecast_run_timestamp=forecast_run_ts,
                    forecast_run_id=forecast_run_id,
                    available_time=forecast_run_ts,
                )
                pops = [
                    r.get("precipitation_probability")
                    for r in hourly
                    if r.get("precipitation_probability") is not None
                ]
                if pops:
                    aggregated["niederschlag_wahrscheinlichkeit"] = round(
                        sum(pops) / len(pops), 2
                    )
                outcome = self._upsert(aggregated, data_type=DAILY_FORECAST_DATA_TYPE)
                if outcome == "inserted":
                    inserted += 1
                else:
                    updated += 1

        self.db.commit()
        ingestion_records_total.labels(source="weather").inc(inserted + updated)
        return {"inserted": inserted, "updated": updated}

    def backfill_history(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        chunk_days: int = 90,
    ) -> dict[str, Any]:
        """Pull DWD observation history for all capitals."""
        start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        inserted = 0
        updated = 0
        errors: list[str] = []

        for city in CITIES:
            cursor = start
            while cursor <= end:
                chunk_end = min(cursor + timedelta(days=max(chunk_days, 1) - 1), end)
                try:
                    hourly = self._fetch_range(
                        city,
                        cursor.strftime("%Y-%m-%d"),
                        (chunk_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    )
                except Exception as exc:
                    ingestion_errors_total.labels(source="weather").inc()
                    errors.append(f"{city['name']} {cursor.date()}..{chunk_end.date()}: {exc}")
                    cursor = chunk_end + timedelta(days=1)
                    continue

                by_day: dict[str, list[dict]] = {}
                for row in hourly:
                    day_key = str(row.get("timestamp") or "")[:10]
                    if not day_key:
                        continue
                    by_day.setdefault(day_key, []).append(row)

                for day_str, rows in by_day.items():
                    if len(rows) < 6:
                        continue
                    day = datetime.strptime(day_str, "%Y-%m-%d")
                    if day < cursor or day > chunk_end:
                        continue
                    record = self._aggregate_day(rows, city=city["name"], day=day)
                    outcome = self._upsert(record, data_type=DAILY_OBSERVATION_DATA_TYPE)
                    if outcome == "inserted":
                        inserted += 1
                    else:
                        updated += 1
                self.db.commit()
                cursor = chunk_end + timedelta(days=1)

        ingestion_records_total.labels(source="weather").inc(inserted + updated)
        return {
            "success": not errors or (inserted + updated) > 0,
            "inserted": inserted,
            "updated": updated,
            "errors": errors[:10] if errors else [],
            "date_range": f"{start.date()}..{end.date()}",
            "cities": len(CITIES),
            "timestamp": utc_now().isoformat(),
        }

    def run_full_import(self, *, include_forecast: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {"source": "BrightSky", "timestamp": utc_now().isoformat()}

        try:
            result["current"] = self.import_current()
        except Exception as exc:
            ingestion_errors_total.labels(source="weather").inc()
            logger.error("Current weather import failed: %s", exc)
            result["current_error"] = str(exc)

        try:
            end = utc_now()
            start = end - timedelta(days=7)
            result["backfill_7d"] = self.backfill_history(start, end)
        except Exception as exc:
            ingestion_errors_total.labels(source="weather").inc()
            logger.error("7-day backfill failed: %s", exc)
            result["backfill_7d_error"] = str(exc)

        if include_forecast:
            try:
                result["forecast"] = self.import_forecast()
            except Exception as exc:
                ingestion_errors_total.labels(source="weather").inc()
                logger.error("Forecast import failed: %s", exc)
                result["forecast_error"] = str(exc)

        result["total_in_db"] = self.db.query(WeatherData).count()
        return result
