"""Response schemas for the outcome API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class OutcomeUploadIssue(BaseModel):
    row: int
    code: str
    message: str


class OutcomeUploadResponse(BaseModel):
    batch_id: str
    filename: str
    brand: str | None
    rows_total: int
    rows_valid: int
    rows_imported: int
    rows_rejected: int
    rows_duplicate: int
    week_min: datetime | None
    week_max: datetime | None
    metrics_seen: list[str]
    regions_seen: list[str]
    issues: list[OutcomeUploadIssue]


class OutcomeSeriesPoint(BaseModel):
    week_start: str
    value: float | None = None
    concentration: float | None = None


class LagCurvePoint(BaseModel):
    lag_days: int
    pearson: float | None


class CorrelationResponse(BaseModel):
    brand: str
    product: str
    region_code: str
    pollen_type: str
    metric: str
    n_weeks: int
    best_lag_days: int
    best_pearson: float
    lift_high_vs_low_pct: float
    high_weeks: int
    low_weeks: int
    lag_curve: list[LagCurvePoint]
    outcome_series: list[OutcomeSeriesPoint]
    pollen_series: list[OutcomeSeriesPoint]


class OutcomeCatalogEntry(BaseModel):
    brand: str
    product: str
    region_code: str
    metric: str
    n_weeks: int
    week_min: datetime | None
    week_max: datetime | None
    source_labels: list[str]


class OutcomeCatalogResponse(BaseModel):
    entries: list[OutcomeCatalogEntry]
