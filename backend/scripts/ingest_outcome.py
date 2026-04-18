"""CLI: ingest a customer outcome CSV (Hexal template, IQVIA-style export).

Usage:
    python -m scripts.ingest_outcome --file path/to/hexal_lorano_2024.csv
    python -m scripts.ingest_outcome --file ... --source-label customer_hexal_pilot
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
from app.services.outcome.upload_service import OutcomeUploadService


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a customer outcome CSV.")
    parser.add_argument("--file", required=True, help="Path to the customer CSV.")
    parser.add_argument(
        "--source-label",
        default="customer_upload",
        help="Label that will be recorded with every row (default: customer_upload).",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Override the auto-generated batch id (useful for deterministic tests).",
    )
    args = parser.parse_args()

    setup_logging(service_name="pollencast-outcome-ingest", environment="cli")
    logger = logging.getLogger("ingest_outcome")

    path = Path(args.file)
    if not path.exists():
        logger.error("File not found: %s", path)
        return 1
    csv_bytes = path.read_bytes()

    db = SessionLocal()
    try:
        service = OutcomeUploadService(db)
        report = service.ingest_csv(
            csv_content=csv_bytes,
            filename=path.name,
            source_label=args.source_label,
            batch_id=args.batch_id,
        )
    finally:
        db.close()

    print(json.dumps(report.to_dict(), indent=2, default=str))
    if report.rows_imported == 0 and report.rows_rejected > 0:
        logger.error("Ingest produced zero accepted rows; fix the CSV and retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
