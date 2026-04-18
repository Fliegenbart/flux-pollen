"""Tests for the customer outcome upload service."""

from __future__ import annotations

import textwrap

import pytest

from app.models.database import OutcomeObservation, UploadHistory
from app.services.outcome.upload_service import OutcomeUploadService


_VALID_CSV = textwrap.dedent(
    """\
    brand,product,region_code,week_start,metric,value
    hexal,lorano_5mg_20stk,BY,2024-04-01,sell_out_units,4321
    hexal,lorano_5mg_20stk,BY,2024-04-08,sell_out_units,4412
    hexal,lorano_5mg_20stk,NW,2024-04-01,sell_out_units,7100
    hexal,lorano_5mg_20stk,NW,2024-04-08,sell_out_units,7390
    hexal,lorano_5mg_20stk,BY,2024-04-01,sell_out_revenue_eur,38447
    """
)


def test_valid_csv_persists_all_rows_and_records_history(db_session):
    service = OutcomeUploadService(db_session)
    report = service.ingest_csv(
        csv_content=_VALID_CSV,
        filename="hexal_lorano_q2_2024.csv",
        source_label="pilot_hexal",
    )
    assert report.rows_total == 5
    assert report.rows_imported == 5
    assert report.rows_rejected == 0
    assert report.rows_duplicate == 0
    assert report.brand == "hexal"
    assert set(report.metrics_seen) == {"sell_out_units", "sell_out_revenue_eur"}
    assert set(report.regions_seen) == {"BY", "NW"}

    stored = db_session.query(OutcomeObservation).count()
    assert stored == 5

    sample = (
        db_session.query(OutcomeObservation)
        .filter_by(region_code="NW", metric_name="sell_out_units")
        .order_by(OutcomeObservation.window_start)
        .all()
    )
    assert [int(row.metric_value) for row in sample] == [7100, 7390]
    assert sample[0].metric_unit == "Packungen"
    assert sample[0].source_label == "pilot_hexal"

    history = db_session.query(UploadHistory).one()
    assert history.upload_type == "outcome_csv"
    assert history.status == "success"
    assert history.summary["rows_imported"] == 5


def test_reingest_same_csv_is_idempotent(db_session):
    service = OutcomeUploadService(db_session)
    first = service.ingest_csv(csv_content=_VALID_CSV, filename="one.csv")
    assert first.rows_imported == 5

    second = service.ingest_csv(csv_content=_VALID_CSV, filename="one.csv")
    assert second.rows_total == 5
    assert second.rows_imported == 0
    assert second.rows_duplicate == 5
    assert db_session.query(OutcomeObservation).count() == 5


def test_unknown_region_is_rejected_with_row_number(db_session):
    csv = textwrap.dedent(
        """\
        brand,product,region_code,week_start,metric,value
        hexal,lorano_5mg_20stk,BY,2024-04-01,sell_out_units,100
        hexal,lorano_5mg_20stk,ZZ,2024-04-08,sell_out_units,200
        """
    )
    report = OutcomeUploadService(db_session).ingest_csv(csv_content=csv, filename="mixed.csv")
    assert report.rows_total == 2
    assert report.rows_imported == 1
    assert report.rows_rejected == 1
    bad = [i for i in report.issues if i.code == "unknown_region"]
    assert bad and bad[0].row_number == 3


def test_missing_required_column_short_circuits(db_session):
    csv = "brand,product,region_code,metric,value\nhexal,lorano_5mg_20stk,BY,sell_out_units,100\n"
    report = OutcomeUploadService(db_session).ingest_csv(csv_content=csv, filename="bad.csv")
    assert report.rows_imported == 0
    assert any(i.code == "missing_columns" for i in report.issues)


def test_unsupported_metric_is_rejected(db_session):
    csv = textwrap.dedent(
        """\
        brand,product,region_code,week_start,metric,value
        hexal,lorano_5mg_20stk,BY,2024-04-01,never_heard_of_this,42
        """
    )
    report = OutcomeUploadService(db_session).ingest_csv(csv_content=csv, filename="bad.csv")
    assert report.rows_imported == 0
    assert any(i.code == "unsupported_metric" for i in report.issues)


def test_negative_values_are_rejected(db_session):
    csv = textwrap.dedent(
        """\
        brand,product,region_code,week_start,metric,value
        hexal,lorano_5mg_20stk,BY,2024-04-01,sell_out_units,-1
        """
    )
    report = OutcomeUploadService(db_session).ingest_csv(csv_content=csv, filename="bad.csv")
    assert report.rows_imported == 0
    assert any(i.code == "negative_value" for i in report.issues)


def test_non_monday_date_is_snapped_to_iso_week_monday(db_session):
    # 2024-04-03 is a Wednesday; should be stored with week_start 2024-04-01 (Mon).
    csv = textwrap.dedent(
        """\
        brand,product,region_code,week_start,metric,value
        hexal,lorano_5mg_20stk,BY,2024-04-03,sell_out_units,111
        """
    )
    report = OutcomeUploadService(db_session).ingest_csv(csv_content=csv, filename="snap.csv")
    assert report.rows_imported == 1
    row = db_session.query(OutcomeObservation).one()
    assert row.window_start.strftime("%Y-%m-%d") == "2024-04-01"
