"""Export a complete demo snapshot for the static frontend.

Writes ``frontend/public/snapshot.json`` containing everything the
dashboard needs to render without a live backend:

- Recent daily-mean pollen series per (pollen, region) + 7/14-day
  forecast produced by the currently-saved model artefact.
- Per-pollen lead-time analysis of the top historical peaks:
  how many days *before* each peak did our model signal vs. the
  persistence and seasonal baselines.
- Pollen × outcome correlation for the ingested Hexal demo data —
  the 78 % high-vs-low lift number lives here.
- Flat backtest table (existing).
- Hero numbers that the page headline reads from.

Re-run any time the local DB changes:

    cd backend
    set -a; source .env.local; set +a
    python -m scripts.export_snapshot
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
import pandas as pd
from sqlalchemy import func

from app.core.logging_config import setup_logging
from app.db.session import SessionLocal
from app.models.database import BacktestRun, OutcomeObservation, PollenObservation
from app.services.data_ingest.region_mapping import BUNDESLAND_NAMES
from app.services.ml.feature_engineering import (
    FeatureBuildConfig,
    assemble_training_frame,
    build_daily_panel,
)
from app.services.ml.lead_time_analysis import (
    compute_lead_time_summary,
    compute_lead_times_for_scope,
)
from app.services.ml.model_registry import load_artifact
from app.services.outcome.correlation_service import OutcomeCorrelationService

FRONTEND_SNAPSHOT = Path(__file__).resolve().parents[2] / "frontend" / "public" / "snapshot.json"
POLLENS = ("birke", "graeser", "erle")
DEFAULT_REGION = "BY"
POLLEN_LABELS = {
    "birke": "Birke",
    "graeser": "Gräser",
    "erle": "Erle",
    "hasel": "Hasel",
    "esche": "Esche",
    "roggen": "Roggen",
    "beifuss": "Beifuß",
    "ambrosia": "Ambrosia",
}
HERO_HEXAL_DEMO = {
    "brand": "hexal",
    "product": "lorano_5mg_20stk",
    "region": DEFAULT_REGION,
    "pollen_type": "birke",
    "metric": "sell_out_units",
}


def _serialize_dt(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).to_pydatetime().isoformat()
    return value


def _pollen_history_and_forecast(db, pollen: str, region: str, end: datetime) -> dict:
    history_start = end - timedelta(days=120)
    rows = (
        db.query(
            func.date(PollenObservation.from_time).label("datum"),
            func.avg(PollenObservation.concentration).label("concentration"),
        )
        .filter(
            PollenObservation.region_code == region,
            PollenObservation.pollen_type == pollen,
            PollenObservation.from_time >= history_start,
            PollenObservation.from_time <= end,
        )
        .group_by(func.date(PollenObservation.from_time))
        .order_by(func.date(PollenObservation.from_time))
        .all()
    )
    history = [
        {"date": str(row[0])[:10], "concentration": round(float(row[1] or 0.0), 2)}
        for row in rows
    ]

    forecasts = []
    for h in (7, 14):
        try:
            art = load_artifact(pollen_type=pollen, region_code=region, horizon_days=h)
        except FileNotFoundError:
            continue
        feat_cfg = FeatureBuildConfig(
            region_code=region,
            pollen_type=pollen,
            start_date=end - timedelta(days=730),
            end_date=end,
        )
        panel = build_daily_panel(db, feat_cfg)
        X, _y, idx = assemble_training_frame(panel, horizon=h)
        if X.empty:
            continue
        X = X[art.metadata.feature_columns]
        last_row = X.iloc[[-1]]
        out = art.service.predict(last_row)
        forecast_date = idx[-1]
        target_date = forecast_date + timedelta(days=h)
        forecasts.append(
            {
                "horizon_days": h,
                "forecast_date": forecast_date.to_pydatetime().isoformat(),
                "target_date": target_date.to_pydatetime().isoformat(),
                "predicted": round(float(out.predicted[0]), 2),
                "lower": round(float(out.lower[0]), 2),
                "upper": round(float(out.upper[0]), 2),
                "model_version": art.metadata.model_version,
            }
        )
    return {"history": history, "forecasts": forecasts}


def _backtest_rows(db) -> list[dict]:
    runs = db.query(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(30).all()
    rows = []
    for r in runs:
        metrics = r.metrics or {}
        improvements = r.improvement_vs_baselines or {}
        rows.append(
            {
                "run_id": r.run_id,
                "pollen_type": r.pollen_type,
                "region_code": r.region_code,
                "horizon_days": r.horizon_days,
                "folds": r.chart_points or 0,
                "model_version": r.model_version,
                "mae": round(float(metrics.get("mae", 0) or 0), 2),
                "wis80": round(float(metrics.get("wis80", 0) or 0), 2),
                "coverage80": round(float(metrics.get("coverage80", 0) or 0), 3),
                "improvement_vs_persistence": round(
                    float(improvements.get("persistence", 0) or 0), 4
                ),
                "improvement_vs_seasonal_naive": round(
                    float(improvements.get("seasonal_naive", 0) or 0), 4
                ),
            }
        )
    return rows


def _lead_time_story(db, pollen: str, region: str) -> dict | None:
    # Pool all recent runs for the scope (one per horizon). That way each
    # peak has multiple fold predictions at different horizons and the
    # model can genuinely "first signal" before the 7-day horizon.
    peaks = compute_lead_times_for_scope(
        db,
        pollen_type=pollen,
        region_code=region,
        horizon_days=None,
        model_version_prefix="pollencast-stacked",
        top_n_peaks=10,
    )
    summary = compute_lead_time_summary(peaks)
    return {
        "scope": {"pollen_type": pollen, "region_code": region, "horizon_days": "all"},
        "summary": summary,
        "peaks": [
            {
                "target_date": p.target_date.isoformat(),
                "observed_value": p.observed_value,
                "event_threshold": p.event_threshold,
                "model_first_signal": _serialize_dt(p.model_first_signal),
                "model_lead_days": p.model_lead_days,
                "persistence_first_signal": _serialize_dt(p.persistence_first_signal),
                "persistence_lead_days": p.persistence_lead_days,
                "seasonal_first_signal": _serialize_dt(p.seasonal_first_signal),
                "seasonal_lead_days": p.seasonal_lead_days,
                "folds_considered": p.folds_considered,
            }
            for p in peaks
        ],
    }


def _correlation_story(db) -> dict | None:
    service = OutcomeCorrelationService(db)
    try:
        result = service.compute(
            brand=HERO_HEXAL_DEMO["brand"],
            product=HERO_HEXAL_DEMO["product"],
            region_code=HERO_HEXAL_DEMO["region"],
            pollen_type=HERO_HEXAL_DEMO["pollen_type"],
            metric=HERO_HEXAL_DEMO["metric"],
        )
    except ValueError:
        return None
    return {
        "brand": result.brand,
        "product": result.product,
        "region_code": result.region_code,
        "pollen_type": result.pollen_type,
        "metric": result.metric,
        "n_weeks": result.n_weeks,
        "best_lag_days": result.best_lag_days,
        "best_pearson": round(result.best_pearson, 3),
        "lift_high_vs_low_pct": result.lift_high_vs_low_pct,
        "high_weeks": result.high_weeks,
        "low_weeks": result.low_weeks,
        "lag_curve": result.lag_curve,
        "outcome_series": result.outcome_series[-60:],
        "pollen_series": result.pollen_series[-60:],
    }


def _hero_numbers(backtests: list[dict], lead_time: dict | None, correlation: dict | None) -> dict:
    # Pull the stacked+cp backtest rows to compute pitch headlines.
    stacked_cp = [b for b in backtests if "cp80" in (b.get("model_version") or "")]
    avg_wis_vs_pers = (
        round(float(np.mean([b["improvement_vs_persistence"] for b in stacked_cp])) * 100.0, 1)
        if stacked_cp
        else None
    )
    avg_wis_vs_seas = (
        round(float(np.mean([b["improvement_vs_seasonal_naive"] for b in stacked_cp])) * 100.0, 1)
        if stacked_cp
        else None
    )
    avg_coverage = (
        round(float(np.mean([b["coverage80"] for b in stacked_cp])), 2) if stacked_cp else None
    )
    model_lead = None
    pers_lead = None
    if lead_time and lead_time.get("summary", {}).get("model_lead_days_mean") is not None:
        model_lead = lead_time["summary"]["model_lead_days_mean"]
        pers_lead = lead_time["summary"].get("persistence_lead_days_median")
    correlation_lift = correlation["lift_high_vs_low_pct"] if correlation else None
    correlation_pearson = correlation["best_pearson"] if correlation else None
    correlation_lag = correlation["best_lag_days"] if correlation else None

    return {
        "wis_improvement_vs_persistence_pct": avg_wis_vs_pers,
        "wis_improvement_vs_seasonal_pct": avg_wis_vs_seas,
        "coverage80_avg": avg_coverage,
        "correlation_lift_high_vs_low_pct": correlation_lift,
        "correlation_best_pearson": correlation_pearson,
        "correlation_best_lag_days": correlation_lag,
        "lead_time_model_days_mean": model_lead,
        "lead_time_persistence_days_median": pers_lead,
    }


def main() -> int:
    setup_logging(service_name="pollencast-snapshot", environment="cli")
    logger = logging.getLogger("export_snapshot")

    end = datetime(2026, 4, 17)
    db = SessionLocal()
    try:
        pollens = {}
        for p in POLLENS:
            pollens[p] = _pollen_history_and_forecast(db, p, DEFAULT_REGION, end)

        backtests = _backtest_rows(db)
        lead_time = _lead_time_story(db, "birke", DEFAULT_REGION)
        correlation = _correlation_story(db)

        snapshot = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "region_code": DEFAULT_REGION,
            "region_name": BUNDESLAND_NAMES[DEFAULT_REGION],
            "hero": _hero_numbers(backtests, lead_time, correlation),
            "pollens": pollens,
            "backtests": backtests,
            "lead_time": lead_time,
            "correlation": correlation,
            "pollen_labels": POLLEN_LABELS,
            "disclaimer": (
                "Korrelations-Daten stammen aus einem synthetischen Hexal-Demo-"
                "Datensatz, der an die echten ePIN-Bayern-Pollen-Daten gekoppelt "
                "ist. Der MOAT (reale Kunden-Elastizität) entsteht erst mit "
                "echten Sell-Out-Daten aus einem Pilot."
            ),
        }
    finally:
        db.close()

    FRONTEND_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    FRONTEND_SNAPSHOT.write_text(json.dumps(snapshot, indent=2, default=str))
    logger.info("Wrote snapshot: %s (%s KB)", FRONTEND_SNAPSHOT, FRONTEND_SNAPSHOT.stat().st_size // 1024)
    print(json.dumps({"ok": True, "path": str(FRONTEND_SNAPSHOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
