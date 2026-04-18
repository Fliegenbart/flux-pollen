"""CLI entry point: ingest BrightSky weather (current + 7d backfill + forecast).

Usage:
    python -m scripts.run_weather_ingest
    python -m scripts.run_weather_ingest --no-forecast
    python -m scripts.run_weather_ingest --backfill-from 2024-01-01 --backfill-to 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logging_config import setup_logging
from app.db.session import SessionLocal
from app.services.data_ingest.weather_service import WeatherService


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest BrightSky weather data")
    parser.add_argument("--no-forecast", action="store_true", help="Skip MOSMIX forecast import")
    parser.add_argument("--backfill-from", type=_parse_date, help="Start date for backfill (YYYY-MM-DD)")
    parser.add_argument("--backfill-to", type=_parse_date, help="End date for backfill (YYYY-MM-DD)")
    args = parser.parse_args()

    setup_logging(service_name="pollencast-ingest", environment="cli")
    logger = logging.getLogger("run_weather_ingest")

    db = SessionLocal()
    try:
        service = WeatherService(db)
        if args.backfill_from and args.backfill_to:
            result = service.backfill_history(args.backfill_from, args.backfill_to)
        else:
            result = service.run_full_import(include_forecast=not args.no_forecast)
    finally:
        db.close()

    print(json.dumps(result, default=str, indent=2))
    logger.info("Weather ingest done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
