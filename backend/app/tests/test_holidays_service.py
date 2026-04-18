"""Holiday import tests against a synthetic schulferien-api.de payload."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from app.models.database import SchoolHolidays
from app.services.data_ingest.holidays_service import SchoolHolidaysService


_PAYLOAD_2026 = [
    {
        "name": "osterferien",
        "name_cp": "Osterferien",
        "start": "2026-04-07T00:00:00",
        "end": "2026-04-18T00:00:00",
        "year": 2026,
        "stateCode": "NW",
        "slug": "osterferien-2026-nordrhein-westfalen",
    },
    {
        "name": "sommerferien",
        "name_cp": "Sommerferien",
        "start": "2026-07-06T00:00:00",
        "end": "2026-08-18T00:00:00",
        "year": 2026,
        "stateCode": "BY",
        "slug": "sommerferien-2026-bayern",
    },
    {
        # Foreign code must be skipped, not crash.
        "name": "herbstferien",
        "name_cp": "Herbstferien",
        "start": "2026-10-12T00:00:00",
        "end": "2026-10-23T00:00:00",
        "year": 2026,
        "stateCode": "XX",
        "slug": "herbstferien-2026-unknown",
    },
]


def test_import_year_inserts_known_states_skips_unknown(db_session):
    service = SchoolHolidaysService(db_session)
    with patch.object(service, "fetch_year", return_value=_PAYLOAD_2026):
        inserted = service.import_year(2026)

    assert inserted == 2
    rows = db_session.query(SchoolHolidays).all()
    assert {row.bundesland for row in rows} == {"NW", "BY"}


def test_import_year_is_idempotent(db_session):
    service = SchoolHolidaysService(db_session)
    with patch.object(service, "fetch_year", return_value=_PAYLOAD_2026):
        service.import_year(2026)
        inserted_second = service.import_year(2026)

    assert inserted_second == 0
    assert db_session.query(SchoolHolidays).count() == 2


def test_import_year_updates_end_date_when_revised(db_session):
    service = SchoolHolidaysService(db_session)
    with patch.object(service, "fetch_year", return_value=_PAYLOAD_2026):
        service.import_year(2026)

    revised = [dict(entry) for entry in _PAYLOAD_2026]
    revised[0]["end"] = "2026-04-19T00:00:00"

    with patch.object(service, "fetch_year", return_value=revised):
        service.import_year(2026)

    nw = (
        db_session.query(SchoolHolidays)
        .filter(SchoolHolidays.bundesland == "NW")
        .one()
    )
    assert nw.end_datum == datetime(2026, 4, 19, 0, 0)


def test_is_holiday_respects_state_filter(db_session):
    service = SchoolHolidaysService(db_session)
    with patch.object(service, "fetch_year", return_value=_PAYLOAD_2026):
        service.import_year(2026)

    assert service.is_holiday(datetime(2026, 4, 10), bundesland="NW") is True
    assert service.is_holiday(datetime(2026, 4, 10), bundesland="BY") is False
    assert service.is_holiday(datetime(2026, 7, 20), bundesland="BY") is True
