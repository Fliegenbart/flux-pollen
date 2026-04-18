"""Customer outcome-data uploader.

Reads a long-format CSV (one metric per row, Bundesland × week), runs
strict validation, and upserts into ``outcome_observations``. Returns a
structured report with per-row issues so an analyst on the customer
side can fix and re-upload without any back-and-forth with us.

Validation is deliberately loud: any row with an unknown region, an
unsupported metric, a negative value or a malformed date is rejected
with a precise location. The alternative — silent drops or guesses —
corrupts the correlation analysis downstream and destroys customer
trust the first time they find a mismatch.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

import pandas as pd
from sqlalchemy.orm import Session

from app.core.time import utc_now
from app.models.database import OutcomeObservation, UploadHistory
from app.services.outcome.schemas import (
    METRICS,
    OPTIONAL_CSV_COLUMNS,
    REQUIRED_CSV_COLUMNS,
    SUPPORTED_METRICS,
    SUPPORTED_REGIONS,
    MetricDefinition,
)

logger = logging.getLogger(__name__)

__all__ = ["OutcomeUploadService", "UploadReport", "UploadIssue"]


@dataclass(frozen=True)
class UploadIssue:
    row_number: int
    code: str
    message: str


@dataclass
class UploadReport:
    batch_id: str
    filename: str
    brand: str | None
    rows_total: int = 0
    rows_valid: int = 0
    rows_imported: int = 0
    rows_rejected: int = 0
    rows_duplicate: int = 0
    week_min: datetime | None = None
    week_max: datetime | None = None
    metrics_seen: list[str] = field(default_factory=list)
    regions_seen: list[str] = field(default_factory=list)
    issues: list[UploadIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "filename": self.filename,
            "brand": self.brand,
            "rows_total": self.rows_total,
            "rows_valid": self.rows_valid,
            "rows_imported": self.rows_imported,
            "rows_rejected": self.rows_rejected,
            "rows_duplicate": self.rows_duplicate,
            "week_min": self.week_min.isoformat() if self.week_min else None,
            "week_max": self.week_max.isoformat() if self.week_max else None,
            "metrics_seen": sorted(set(self.metrics_seen)),
            "regions_seen": sorted(set(self.regions_seen)),
            "issues": [
                {"row": issue.row_number, "code": issue.code, "message": issue.message}
                for issue in self.issues
            ],
        }


class OutcomeUploadService:
    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #

    def ingest_csv(
        self,
        *,
        csv_content: str | bytes,
        filename: str,
        source_label: str = "customer_upload",
        batch_id: str | None = None,
    ) -> UploadReport:
        batch_id = batch_id or f"outcome_{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
        report = UploadReport(batch_id=batch_id, filename=filename, brand=None)

        frame = self._load_frame(csv_content, report=report)
        if frame is None:
            self._write_upload_history(report, status="error")
            return report

        valid_rows = self._validate(frame, report=report)
        self._persist(valid_rows, source_label=source_label, batch_id=batch_id, report=report)
        self._write_upload_history(
            report,
            status="error" if report.rows_imported == 0 and report.rows_rejected else "success",
        )
        return report

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def _load_frame(self, raw: str | bytes, *, report: UploadReport) -> pd.DataFrame | None:
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = raw
        try:
            frame = pd.read_csv(io.StringIO(text))
        except Exception as exc:
            report.issues.append(
                UploadIssue(row_number=0, code="csv_parse_error", message=str(exc))
            )
            return None

        missing = [col for col in REQUIRED_CSV_COLUMNS if col not in frame.columns]
        if missing:
            report.issues.append(
                UploadIssue(
                    row_number=0,
                    code="missing_columns",
                    message=f"Required columns missing: {missing}",
                )
            )
            return None
        for col in OPTIONAL_CSV_COLUMNS:
            if col not in frame.columns:
                frame[col] = None

        report.rows_total = int(len(frame))
        return frame

    def _validate(
        self,
        frame: pd.DataFrame,
        *,
        report: UploadReport,
    ) -> list[dict[str, Any]]:
        valid: list[dict[str, Any]] = []
        brands: set[str] = set()
        metrics: set[str] = set()
        regions: set[str] = set()
        week_min: datetime | None = None
        week_max: datetime | None = None

        for index, row in frame.iterrows():
            row_number = int(index) + 2  # header counts as row 1 in a spreadsheet
            brand = str(row.get("brand") or "").strip().lower()
            product = str(row.get("product") or "").strip().lower()
            region_code = str(row.get("region_code") or "").strip().upper()
            metric = str(row.get("metric") or "").strip().lower()
            raw_value = row.get("value")
            week_start_raw = row.get("week_start")

            if not brand or not product:
                report.issues.append(
                    UploadIssue(row_number, "missing_brand_or_product", "brand and product must be set")
                )
                continue
            if metric not in SUPPORTED_METRICS:
                report.issues.append(
                    UploadIssue(
                        row_number,
                        "unsupported_metric",
                        f"Unknown metric {metric!r}. Allowed: {sorted(SUPPORTED_METRICS)}.",
                    )
                )
                continue
            if region_code not in SUPPORTED_REGIONS:
                report.issues.append(
                    UploadIssue(
                        row_number,
                        "unknown_region",
                        f"region_code {region_code!r} is not one of the 16 Bundesländer.",
                    )
                )
                continue
            try:
                week_start = _parse_week_start(week_start_raw)
            except ValueError as exc:
                report.issues.append(UploadIssue(row_number, "bad_week_start", str(exc)))
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                report.issues.append(
                    UploadIssue(row_number, "non_numeric_value", f"value={raw_value!r} is not numeric")
                )
                continue
            if value < 0:
                report.issues.append(
                    UploadIssue(row_number, "negative_value", f"value={value} < 0 is not allowed")
                )
                continue

            metric_def = METRICS[metric]
            channel = row.get("channel")
            campaign_id = row.get("campaign_id")

            valid.append(
                {
                    "brand": brand,
                    "product": product,
                    "region_code": region_code,
                    "metric": metric,
                    "metric_def": metric_def,
                    "week_start": week_start,
                    "week_end": week_start + timedelta(days=7),
                    "value": value,
                    "channel": str(channel).strip() if channel and str(channel).strip() else None,
                    "campaign_id": str(campaign_id).strip() if campaign_id and str(campaign_id).strip() else None,
                }
            )
            brands.add(brand)
            metrics.add(metric)
            regions.add(region_code)
            week_min = week_start if week_min is None or week_start < week_min else week_min
            week_max = week_start if week_max is None or week_start > week_max else week_max

        report.rows_valid = len(valid)
        report.rows_rejected = report.rows_total - report.rows_valid
        report.metrics_seen = list(metrics)
        report.regions_seen = list(regions)
        report.week_min = week_min
        report.week_max = week_max
        if brands:
            # If a file mixes brands we record them all but flag it.
            report.brand = sorted(brands)[0] if len(brands) == 1 else "+".join(sorted(brands))
            if len(brands) > 1:
                report.issues.append(
                    UploadIssue(
                        0,
                        "multi_brand_upload",
                        f"File contains multiple brands: {sorted(brands)}. "
                        "Downstream correlation analysis will still work, but split "
                        "into per-brand files for cleaner reporting.",
                    )
                )
        return valid

    def _persist(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        source_label: str,
        batch_id: str,
        report: UploadReport,
    ) -> None:
        for row in rows:
            metric_def: MetricDefinition = row["metric_def"]
            existing = (
                self.db.query(OutcomeObservation)
                .filter(
                    OutcomeObservation.brand == row["brand"],
                    OutcomeObservation.product == row["product"],
                    OutcomeObservation.region_code == row["region_code"],
                    OutcomeObservation.metric_name == row["metric"],
                    OutcomeObservation.window_start == row["week_start"],
                    OutcomeObservation.window_end == row["week_end"],
                    OutcomeObservation.source_label == source_label,
                )
                .one_or_none()
            )
            if existing is not None:
                existing.metric_value = row["value"]
                existing.metric_unit = metric_def.unit
                existing.channel = row["channel"]
                existing.campaign_id = row["campaign_id"]
                existing.metadata_json = {
                    "batch_id": batch_id,
                    "metric_group": metric_def.group,
                }
                report.rows_duplicate += 1
            else:
                self.db.add(
                    OutcomeObservation(
                        brand=row["brand"],
                        product=row["product"],
                        region_code=row["region_code"],
                        window_start=row["week_start"],
                        window_end=row["week_end"],
                        metric_name=row["metric"],
                        metric_value=row["value"],
                        metric_unit=metric_def.unit,
                        source_label=source_label,
                        channel=row["channel"],
                        campaign_id=row["campaign_id"],
                        metadata_json={
                            "batch_id": batch_id,
                            "metric_group": metric_def.group,
                        },
                    )
                )
                report.rows_imported += 1
        self.db.commit()

    def _write_upload_history(self, report: UploadReport, *, status: str) -> None:
        entry = UploadHistory(
            filename=report.filename,
            upload_type="outcome_csv",
            file_format="csv",
            row_count=report.rows_total,
            date_range_start=report.week_min,
            date_range_end=report.week_max,
            status=status,
            error_message=(
                report.issues[0].message if report.issues and status == "error" else None
            ),
            summary=report.to_dict(),
        )
        self.db.add(entry)
        self.db.commit()


def _parse_week_start(value: Any) -> datetime:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("week_start is empty")
    if isinstance(value, (datetime, pd.Timestamp)):
        ts = pd.Timestamp(value).to_pydatetime()
    else:
        text = str(value).strip()
        try:
            ts = datetime.fromisoformat(text[:10])
        except ValueError as exc:
            raise ValueError(f"week_start={value!r} is not an ISO date") from exc
    # Customers will often send a Monday; we accept any day and snap to
    # Monday of that ISO week so downstream joins are stable.
    ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    if ts.weekday() != 0:
        ts = ts - timedelta(days=ts.weekday())
    return ts
