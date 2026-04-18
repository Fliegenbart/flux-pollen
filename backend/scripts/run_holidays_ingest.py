"""CLI entry point: ingest German school holidays for the 16 Bundesländer.

Usage:
    python -m scripts.run_holidays_ingest
    python -m scripts.run_holidays_ingest --years 2024 2025 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logging_config import setup_logging
from app.db.session import SessionLocal
from app.services.data_ingest.holidays_service import SchoolHolidaysService


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest German school holidays (schulferien-api.de v2)")
    parser.add_argument("--years", nargs="+", type=int, help="Explicit years (default: prev, current, next)")
    args = parser.parse_args()

    setup_logging(service_name="pollencast-ingest", environment="cli")
    logger = logging.getLogger("run_holidays_ingest")

    db = SessionLocal()
    try:
        service = SchoolHolidaysService(db)
        result = service.run_full_import(years=args.years)
    finally:
        db.close()

    print(json.dumps(result, default=str, indent=2))
    if not result.get("success"):
        logger.warning("Holidays ingest completed with errors.")
        return 1
    logger.info("Holidays ingest OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
