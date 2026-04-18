"""Outcome upload and pollen/outcome correlation endpoints.

Intentionally small surface area:

- ``POST /upload`` — accept a CSV upload, return the structured
  validation report. Rejects on wrong content-type or >5 MB; anything
  beyond that belongs in a dedicated ingest pipeline, not the API.
- ``GET /catalog`` — list everything we have on file, so the frontend
  can populate its brand/product/metric/region selectors.
- ``GET /correlation`` — compute the Pollen × outcome correlation,
  best lag, and high-vs-low lift for a single scope.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.database import OutcomeObservation
from app.schemas.outcome import (
    CorrelationResponse,
    LagCurvePoint,
    OutcomeCatalogEntry,
    OutcomeCatalogResponse,
    OutcomeSeriesPoint,
    OutcomeUploadIssue,
    OutcomeUploadResponse,
)
from app.services.outcome.correlation_service import OutcomeCorrelationService
from app.services.outcome.upload_service import OutcomeUploadService

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@router.post(
    "/upload",
    response_model=OutcomeUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_outcome_csv(
    file: UploadFile = File(...),
    source_label: str = Query(
        "customer_upload",
        min_length=3,
        max_length=64,
        description="Free-form label that stays with every persisted row.",
    ),
    db: Session = Depends(get_db),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large ({len(raw):,} B > {MAX_UPLOAD_BYTES:,} B).",
        )
    service = OutcomeUploadService(db)
    report = service.ingest_csv(
        csv_content=raw,
        filename=file.filename or "upload.csv",
        source_label=source_label,
    )
    return OutcomeUploadResponse(
        batch_id=report.batch_id,
        filename=report.filename,
        brand=report.brand,
        rows_total=report.rows_total,
        rows_valid=report.rows_valid,
        rows_imported=report.rows_imported,
        rows_rejected=report.rows_rejected,
        rows_duplicate=report.rows_duplicate,
        week_min=report.week_min,
        week_max=report.week_max,
        metrics_seen=report.metrics_seen,
        regions_seen=report.regions_seen,
        issues=[
            OutcomeUploadIssue(row=issue.row_number, code=issue.code, message=issue.message)
            for issue in report.issues
        ],
    )


@router.get("/catalog", response_model=OutcomeCatalogResponse)
def outcome_catalog(db: Session = Depends(get_db)):
    rows = (
        db.query(
            OutcomeObservation.brand,
            OutcomeObservation.product,
            OutcomeObservation.region_code,
            OutcomeObservation.metric_name,
            func.count().label("n"),
            func.min(OutcomeObservation.window_start).label("week_min"),
            func.max(OutcomeObservation.window_start).label("week_max"),
        )
        .group_by(
            OutcomeObservation.brand,
            OutcomeObservation.product,
            OutcomeObservation.region_code,
            OutcomeObservation.metric_name,
        )
        .all()
    )
    # Collect the distinct source labels per scope so the UI can show them.
    label_rows = (
        db.query(
            OutcomeObservation.brand,
            OutcomeObservation.product,
            OutcomeObservation.region_code,
            OutcomeObservation.metric_name,
            OutcomeObservation.source_label,
        )
        .distinct()
        .all()
    )
    label_index: dict[tuple[str, str, str, str], list[str]] = {}
    for brand, product, region, metric, label in label_rows:
        label_index.setdefault((brand, product, region, metric), []).append(label)

    entries = [
        OutcomeCatalogEntry(
            brand=row.brand,
            product=row.product,
            region_code=row.region_code,
            metric=row.metric_name,
            n_weeks=int(row.n),
            week_min=row.week_min,
            week_max=row.week_max,
            source_labels=sorted(
                set(label_index.get((row.brand, row.product, row.region_code, row.metric_name), []))
            ),
        )
        for row in rows
    ]
    entries.sort(key=lambda e: (e.brand, e.product, e.region_code, e.metric))
    return OutcomeCatalogResponse(entries=entries)


@router.get("/correlation", response_model=CorrelationResponse)
def outcome_correlation(
    brand: str = Query(..., min_length=1),
    product: str = Query(..., min_length=1),
    region: str = Query(..., min_length=2, max_length=2),
    pollen_type: str = Query(...),
    metric: str = Query("sell_out_units"),
    max_lag_days: int = Query(21, ge=0, le=60),
    db: Session = Depends(get_db),
):
    service = OutcomeCorrelationService(db)
    try:
        result = service.compute(
            brand=brand.lower(),
            product=product.lower(),
            region_code=region.upper(),
            pollen_type=pollen_type.lower(),
            metric=metric.lower(),
            lag_days_range=range(-max_lag_days, max_lag_days + 1, 1),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return CorrelationResponse(
        brand=result.brand,
        product=result.product,
        region_code=result.region_code,
        pollen_type=result.pollen_type,
        metric=result.metric,
        n_weeks=result.n_weeks,
        best_lag_days=result.best_lag_days,
        best_pearson=result.best_pearson,
        lift_high_vs_low_pct=result.lift_high_vs_low_pct,
        high_weeks=result.high_weeks,
        low_weeks=result.low_weeks,
        lag_curve=[LagCurvePoint(**item) for item in result.lag_curve],
        outcome_series=[OutcomeSeriesPoint(**item) for item in result.outcome_series],
        pollen_series=[OutcomeSeriesPoint(**item) for item in result.pollen_series],
    )
