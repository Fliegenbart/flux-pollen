"""CLI: train a ForecastService and persist it as a model artefact.

Trains the same Ridge-+-quantile-GBM stack used in the backtester on
the full available history and saves the fitted pipeline to

    backend/app/ml_models/<pollen>/<region>/h<horizon>/model.joblib
    backend/app/ml_models/<pollen>/<region>/h<horizon>/metadata.json

The API endpoint loads these artefacts lazily at request time.

Usage:
    python -m scripts.run_train --pollen birke --region BY --horizon 7
    python -m scripts.run_train --pollen graeser --region BY --horizon 7 --from 2021-01-01
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
from app.services.ml.feature_engineering import (
    FeatureBuildConfig,
    assemble_training_frame,
    build_daily_panel,
)
from app.services.ml.forecast_service import ForecastService
from app.services.ml.model_registry import (
    ModelArtifact,
    build_metadata,
    save_artifact,
)

MODEL_VERSION = "pollencast-ridge-gbm-v0"


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a pollen forecast model")
    parser.add_argument("--pollen", required=True)
    parser.add_argument("--region", default="BY")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--from", dest="from_date", type=_parse_date)
    parser.add_argument("--to", dest="to_date", type=_parse_date)
    args = parser.parse_args()

    setup_logging(service_name="pollencast-train", environment="cli")
    logger = logging.getLogger("run_train")

    to_date = args.to_date or utc_now()
    from_date = args.from_date or (to_date - timedelta(days=3 * 365))

    feature_config = FeatureBuildConfig(
        region_code=args.region,
        pollen_type=args.pollen,
        start_date=from_date,
        end_date=to_date,
    )

    db = SessionLocal()
    try:
        panel = build_daily_panel(db, feature_config)
    finally:
        db.close()
    if panel.empty:
        logger.error("No panel data. Run the ingest scripts first.")
        return 1

    X, y, index = assemble_training_frame(panel, horizon=args.horizon)
    if len(index) < 30:
        logger.error(
            "Only %s training samples available for %s %s h=%s — refusing to train.",
            len(index), args.pollen, args.region, args.horizon,
        )
        return 1

    service = ForecastService()
    service.fit(X, y)

    metadata = build_metadata(
        service=service,
        feature_config=feature_config,
        horizon_days=args.horizon,
        feature_columns=list(X.columns),
        train_n_samples=len(index),
        model_version=MODEL_VERSION,
    )
    artefact = ModelArtifact(service=service, metadata=metadata)
    path = save_artifact(artefact)

    print(json.dumps({
        "success": True,
        "path": str(path),
        "pollen_type": metadata.pollen_type,
        "region_code": metadata.region_code,
        "horizon_days": metadata.horizon_days,
        "train_n_samples": metadata.train_n_samples,
        "training_window": {
            "start": metadata.training_window_start,
            "end": metadata.training_window_end,
        },
        "feature_column_count": len(metadata.feature_columns),
        "model_version": metadata.model_version,
    }, indent=2))
    logger.info("Model artefact saved to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
