"""CLI entry point: ingest the current DWD pollen payload.

Usage:
    python -m scripts.run_pollen_ingest
    python -m scripts.run_pollen_ingest --url https://example.com/s31fg.json
    python -m scripts.run_pollen_ingest --file ./fixtures/s31fg.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow `python scripts/run_pollen_ingest.py` from repo root.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logging_config import setup_logging
from app.db.session import SessionLocal
from app.services.data_ingest.pollen_service import PollenService


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest DWD pollen payload")
    parser.add_argument("--url", help="Override the DWD_POLLEN_URL from config")
    parser.add_argument("--file", help="Ingest a local JSON payload instead of fetching")
    args = parser.parse_args()

    setup_logging(service_name="pollencast-ingest", environment="cli")
    logger = logging.getLogger("run_pollen_ingest")

    db = SessionLocal()
    try:
        service = PollenService(db)
        if args.file:
            result = service.import_from_file(args.file)
        else:
            result = service.run_full_import(source_url=args.url)
    finally:
        db.close()

    print(json.dumps(result, default=str, indent=2))
    if not result.get("success"):
        logger.error("Pollen ingest did not persist any records.")
        return 1
    logger.info("Pollen ingest OK: %s inserted, %s updated", result.get("inserted"), result.get("updated"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
