"""Pollen forecast and observation endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.time import utc_now
from app.db.session import get_db
from app.models.database import PollenData, PollenObservation
from app.schemas.pollen import (
    PollenCurrentResponse,
    PollenForecastPoint,
    PollenForecastResponse,
    RegionalRankingEntry,
    RegionalRankingResponse,
)
from app.services.data_ingest.region_mapping import (
    ALL_BUNDESLAENDER,
    BUNDESLAND_NAMES,
)
from app.services.ml.feature_engineering import (
    FeatureBuildConfig,
    assemble_training_frame,
    build_daily_panel,
)
from app.services.ml.model_registry import load_artifact, list_artifacts

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_HORIZONS: tuple[int, ...] = (7, 14)
SUPPORTED_POLLEN_TYPES: tuple[str, ...] = (
    "hasel", "erle", "esche", "birke", "graeser", "roggen", "beifuss", "ambrosia",
)


def _confidence_label(lower: float, upper: float, predicted: float) -> str:
    """Crude band-width heuristic: tighter relative width ↦ higher confidence."""
    if predicted <= 0:
        return "low"
    width = max(upper - lower, 0.0)
    ratio = width / max(predicted, 1.0)
    if ratio < 0.4:
        return "high"
    if ratio < 1.0:
        return "medium"
    return "low"


@router.get(
    "/current",
    response_model=PollenCurrentResponse,
    tags=["pollen"],
)
def get_current(
    region: str = Query(..., min_length=2, max_length=2, description="Bundesland-Code"),
    pollen_type: str = Query(..., description="e.g. birke, graeser"),
    db: Session = Depends(get_db),
):
    region = region.upper()
    if region not in ALL_BUNDESLAENDER:
        raise HTTPException(status_code=404, detail=f"Unknown region: {region}")

    # Prefer ePIN concentration (richer unit) → fall back to DWD index.
    obs = (
        db.query(PollenObservation)
        .filter(
            PollenObservation.region_code == region,
            PollenObservation.pollen_type == pollen_type,
        )
        .order_by(PollenObservation.from_time.desc())
        .first()
    )
    if obs is not None:
        # Take the day's mean across the 8 3h-buckets to report a daily value.
        same_day = (
            db.query(func.avg(PollenObservation.concentration))
            .filter(
                PollenObservation.region_code == region,
                PollenObservation.pollen_type == pollen_type,
                PollenObservation.from_time >= obs.from_time.replace(hour=0, minute=0, second=0, microsecond=0),
                PollenObservation.from_time < obs.from_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1),
            )
            .scalar()
        )
        return PollenCurrentResponse(
            pollen_type=pollen_type,
            region_code=region,
            datum=obs.from_time,
            concentration=float(same_day) if same_day is not None else obs.concentration,
            pollen_index=None,
            available_time=obs.available_time,
            source="ePIN",
        )

    dwd = (
        db.query(PollenData)
        .filter(PollenData.region_code == region, PollenData.pollen_type == pollen_type)
        .order_by(PollenData.datum.desc())
        .first()
    )
    if dwd is None:
        raise HTTPException(
            status_code=404,
            detail=f"No observations for {pollen_type} in {region}",
        )
    return PollenCurrentResponse(
        pollen_type=pollen_type,
        region_code=region,
        datum=dwd.datum,
        concentration=None,
        pollen_index=dwd.pollen_index,
        available_time=dwd.available_time,
        source="DWD",
    )


@router.get(
    "/forecast",
    response_model=PollenForecastResponse,
    tags=["pollen"],
)
def get_forecast(
    region: str = Query(..., min_length=2, max_length=2),
    pollen_type: str = Query(...),
    horizon_days: int = Query(7, description="7 or 14"),
    db: Session = Depends(get_db),
):
    region = region.upper()
    if region not in ALL_BUNDESLAENDER:
        raise HTTPException(status_code=404, detail=f"Unknown region: {region}")
    if horizon_days not in SUPPORTED_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"horizon_days must be one of {SUPPORTED_HORIZONS}",
        )
    if pollen_type not in SUPPORTED_POLLEN_TYPES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown pollen_type: {pollen_type}",
        )

    try:
        artefact = load_artifact(
            pollen_type=pollen_type,
            region_code=region,
            horizon_days=horizon_days,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"No trained model for {pollen_type}/{region}/h{horizon_days}. "
                "Run scripts/run_train.py."
            ),
        )

    meta = artefact.metadata
    end_date = utc_now()
    start_date = end_date - timedelta(days=730)  # 2y window is plenty for feature lags.
    feature_config = FeatureBuildConfig(
        region_code=region,
        pollen_type=pollen_type,
        start_date=start_date,
        end_date=end_date,
    )
    panel = build_daily_panel(db, feature_config)
    if panel.empty:
        raise HTTPException(
            status_code=503,
            detail="Feature panel is empty — have the ingest scripts run?",
        )

    X, _y, index = assemble_training_frame(panel, horizon=horizon_days)
    if X.empty:
        raise HTTPException(
            status_code=503,
            detail="Not enough recent observations to build a feature vector.",
        )

    # Reindex/validate feature columns. The API must refuse to serve a
    # forecast silently if the model was trained on a different feature set.
    missing = [c for c in meta.feature_columns if c not in X.columns]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Feature engineering drift — missing columns: {missing[:5]}…",
        )
    X = X[meta.feature_columns]

    last_row = X.iloc[[-1]]
    forecast_date = index[-1]
    target_date = forecast_date + timedelta(days=horizon_days)

    output = artefact.service.predict(last_row)
    predicted = float(output.predicted[0])
    lower = float(output.lower[0])
    upper = float(output.upper[0])

    return PollenForecastResponse(
        pollen_type=pollen_type,
        region_code=region,
        forecast_date=forecast_date.to_pydatetime(),
        horizon_days=horizon_days,
        model_version=meta.model_version,
        trained_at=meta.trained_at,
        forecast=PollenForecastPoint(
            target_date=target_date.to_pydatetime(),
            horizon_days=horizon_days,
            predicted_concentration=predicted,
            lower_bound=lower,
            upper_bound=upper,
            confidence_label=_confidence_label(lower, upper, predicted),
        ),
    )


@router.get(
    "/forecast/regional",
    response_model=RegionalRankingResponse,
    tags=["pollen"],
)
def get_regional_ranking(
    pollen_type: str = Query(...),
    horizon_days: int = Query(7),
    db: Session = Depends(get_db),
):
    """Rank every region for which we have a trained model, by predicted concentration."""
    if horizon_days not in SUPPORTED_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"horizon_days must be one of {SUPPORTED_HORIZONS}",
        )

    available_artefacts = [
        a for a in list_artifacts()
        if a.pollen_type == pollen_type and int(a.horizon_days) == int(horizon_days)
    ]
    if not available_artefacts:
        raise HTTPException(
            status_code=503,
            detail=f"No trained models for {pollen_type} h{horizon_days}.",
        )

    entries: list[tuple[str, PollenForecastPoint, datetime]] = []
    for meta in available_artefacts:
        try:
            response = get_forecast(
                region=meta.region_code,
                pollen_type=pollen_type,
                horizon_days=horizon_days,
                db=db,
            )
        except HTTPException as exc:
            logger.warning("Regional ranking skipped %s: %s", meta.region_code, exc.detail)
            continue
        entries.append((meta.region_code, response.forecast, response.forecast_date))

    if not entries:
        raise HTTPException(
            status_code=503,
            detail="No regional forecasts could be produced.",
        )

    entries.sort(key=lambda item: item[1].predicted_concentration, reverse=True)
    ranking = [
        RegionalRankingEntry(
            region_code=code,
            region_name=BUNDESLAND_NAMES.get(code, code),
            predicted_concentration=point.predicted_concentration,
            lower_bound=point.lower_bound,
            upper_bound=point.upper_bound,
            rank=i + 1,
        )
        for i, (code, point, _) in enumerate(entries)
    ]
    forecast_date = entries[0][2]
    return RegionalRankingResponse(
        pollen_type=pollen_type,
        horizon_days=horizon_days,
        forecast_date=forecast_date,
        entries=ranking,
    )
