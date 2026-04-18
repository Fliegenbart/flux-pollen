"""CLI entry point: ingest ePIN Bayern station-level pollen measurements.

Usage:
    # Last 7 days
    python -m scripts.run_epin_ingest

    # Explicit window
    python -m scripts.run_epin_ingest --from 2024-03-01 --to 2024-07-31

    # Multi-year historical backfill in 30-day chunks
    python -m scripts.run_epin_ingest --backfill --from 2020-01-01 --to 2025-12-31

    # Ingest a captured JSON fixture
    python -m scripts.run_epin_ingest --file ./snapshots/epin_2024_may.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logging_config import setup_logging
from app.core.time import utc_now
from app.db.session import SessionLocal
from app.services.data_ingest.epin_service import EPINService


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest ePIN Bayern pollen measurements")
    parser.add_argument("--from", dest="from_time", type=_parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_time", type=_parse_date, help="End date (YYYY-MM-DD)")
    parser.add_argument("--backfill", action="store_true", help="Chunked historical backfill mode")
    parser.add_argument("--chunk-days", type=int, default=30, help="Chunk size for --backfill")
    parser.add_argument("--file", help="Ingest a local JSON payload instead of fetching the API")
    parser.add_argument("--api-base", help="Override the ePIN API base URL")
    args = parser.parse_args()

    setup_logging(service_name="pollencast-ingest", environment="cli")
    logger = logging.getLogger("run_epin_ingest")

    db = SessionLocal()
    try:
        service = EPINService(db)
        if args.file:
            result = service.import_from_file(args.file)
        elif args.backfill:
            if not args.from_time or not args.to_time:
                parser.error("--backfill requires --from and --to")
            result = service.backfill_range(
                start=args.from_time,
                end=args.to_time,
                chunk_days=args.chunk_days,
                api_base=args.api_base,
            )
        else:
            to_time = args.to_time or utc_now()
            from_time = args.from_time or (to_time - timedelta(days=7))
            result = service.run_full_import(
                from_time=from_time,
                to_time=to_time,
                api_base=args.api_base,
            )
    finally:
        db.close()

    print(json.dumps(result, default=str, indent=2))
    if not result.get("success"):
        logger.error("ePIN ingest did not persist any records.")
        return 1
    logger.info(
        "ePIN ingest OK: %s inserted, %s updated",
        result.get("inserted"),
        result.get("updated"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
