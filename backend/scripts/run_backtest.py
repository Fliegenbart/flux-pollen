"""CLI: run a walk-forward backtest for a (pollen_type, region, horizon).

Assumes the ingest scripts have been run first so ``pollen_observations``,
``weather_data`` and ``school_holidays`` contain enough history. For
Bayern + ePIN, 1.5+ years of history is a good starting point.

Usage:
    python -m scripts.run_backtest --pollen birke --region BY
    python -m scripts.run_backtest --pollen graeser --region BY --horizon 7 --from 2023-01-01
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
from app.services.ml.backtester import (
    BacktestConfig,
    WalkForwardBacktester,
    persist_backtest_run,
)
from app.services.ml.feature_engineering import FeatureBuildConfig, build_daily_panel
from app.services.ml.conformal_calibrator import ConformalCalibratedForecaster
from app.services.ml.forecast_service import ForecastService
from app.services.ml.stacked_forecast_service import StackedForecastService


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward backtest for a pollen series")
    parser.add_argument("--pollen", required=True, help="birke, graeser, erle, ...")
    parser.add_argument("--region", default="BY", help="Bundesland code (default: BY)")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon in days")
    parser.add_argument("--from", dest="from_date", type=_parse_date, help="Training-window start (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", type=_parse_date, help="Training-window end (YYYY-MM-DD)")
    parser.add_argument("--min-train-days", type=int, default=60, help="Min fold train length")
    parser.add_argument("--step-days", type=int, default=1, help="Walk-forward step")
    parser.add_argument("--no-persist", action="store_true", help="Print only; skip DB writes")
    parser.add_argument(
        "--stacked",
        action="store_true",
        help="Use the Ridge+HW→XGBoost stacked forecaster instead of the single-stage service.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Wrap the chosen forecaster in a Split-Conformal calibrator targeting 0.80 coverage.",
    )
    args = parser.parse_args()

    setup_logging(service_name="pollencast-backtest", environment="cli")
    logger = logging.getLogger("run_backtest")

    to_date = args.to_date or utc_now()
    from_date = args.from_date or (to_date - timedelta(days=3 * 365))

    feature_config = FeatureBuildConfig(
        region_code=args.region,
        pollen_type=args.pollen,
        start_date=from_date,
        end_date=to_date,
    )
    backtest_config = BacktestConfig(
        horizon_days=args.horizon,
        min_train_days=args.min_train_days,
        step_days=args.step_days,
    )
    def _build_base(cfg: BacktestConfig):
        if args.stacked:
            return StackedForecastService(
                horizon_days=cfg.horizon_days,
                lower_quantile=cfg.lower_quantile,
                upper_quantile=cfg.upper_quantile,
            )
        return ForecastService(
            lower_quantile=cfg.lower_quantile,
            upper_quantile=cfg.upper_quantile,
        )

    if args.stacked or args.calibrate:
        def _factory(cfg: BacktestConfig):
            base = _build_base(cfg)
            if args.calibrate:
                return ConformalCalibratedForecaster(base=base, target_coverage=0.80)
            return base

        backtest_config.forecaster_factory = _factory
        tag_parts = ["stacked-hw-ridge-xgb"] if args.stacked else ["ridge-gbm"]
        if args.calibrate:
            tag_parts.append("cp80")
        backtest_config.model_version = "pollencast-" + "-".join(tag_parts) + "-v0"

    db = SessionLocal()
    try:
        panel = build_daily_panel(db, feature_config)
        if panel.empty:
            logger.error("No panel data — run the ingest scripts first.")
            return 1

        backtester = WalkForwardBacktester(
            pollen_type=args.pollen,
            region_code=args.region,
            config=backtest_config,
        )
        result = backtester.run(panel)
        if not args.no_persist:
            persist_backtest_run(db, result, config=backtest_config)
    finally:
        db.close()

    report = {
        "run_id": result.run_id,
        "pollen_type": result.pollen_type,
        "region_code": result.region_code,
        "horizon_days": result.horizon_days,
        "n_folds": result.n_folds,
        "metrics": result.metrics,
        "baseline_metrics": result.baseline_metrics,
        "improvement_vs_baselines": result.improvement_vs_baselines,
    }
    print(json.dumps(report, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
