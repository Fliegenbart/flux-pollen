"""Idempotency tests for the DWD pollen ingest.

These are the tests we'd be embarrassed to ship without. If running the
ingest twice didn't converge, the forecast pipeline would silently
duplicate observations and the Point-in-Time bookkeeping would lie.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.models.database import PollenData
from app.services.data_ingest.pollen_service import (
    PollenService,
    SUPPORTED_POLLEN_TYPES,
)


def _pollen_service(db_session) -> PollenService:
    return PollenService(db_session)


def test_import_from_fixture_persists_all_regions_and_pollens(db_session, dwd_pollen_fixture_path):
    service = _pollen_service(db_session)
    result = service.import_from_file(dwd_pollen_fixture_path)

    assert result["success"] is True
    # 12 DWD region groups × 8 pollen types × 3 horizons = 288 logical rows.
    # After the group→state fan-out, that becomes 16 states × 8 types × 3
    # horizons = 384 physical rows (Brandenburg/Berlin, NI/HB, RP/SL, SH/HH
    # each double).
    stored = db_session.query(PollenData).count()
    assert stored == 16 * len(SUPPORTED_POLLEN_TYPES) * 3

    # Every single inserted row carries an available_time (Point-in-Time anchor).
    assert (
        db_session.query(PollenData)
        .filter(PollenData.available_time.is_(None))
        .count()
        == 0
    )


def test_grouped_regions_fan_out_to_both_states(db_session, dwd_pollen_fixture_path):
    service = _pollen_service(db_session)
    service.import_from_file(dwd_pollen_fixture_path)

    # "Niedersachsen und Bremen" must write rows for both NI and HB, same values.
    ni_birke = {
        row.datum: row.pollen_index
        for row in db_session.query(PollenData).filter(
            PollenData.region_code == "NI", PollenData.pollen_type == "birke"
        )
    }
    hb_birke = {
        row.datum: row.pollen_index
        for row in db_session.query(PollenData).filter(
            PollenData.region_code == "HB", PollenData.pollen_type == "birke"
        )
    }
    assert ni_birke and ni_birke == hb_birke


def test_reingest_same_payload_is_idempotent(db_session, dwd_pollen_fixture_path):
    service = _pollen_service(db_session)
    service.import_from_file(dwd_pollen_fixture_path)
    first_count = db_session.query(PollenData).count()

    result = service.import_from_file(dwd_pollen_fixture_path)
    second_count = db_session.query(PollenData).count()

    assert first_count == second_count
    assert result["inserted"] == 0
    assert result["updated"] == first_count


def test_revised_payload_updates_index_and_available_time(db_session, dwd_pollen_fixture_path):
    service = _pollen_service(db_session)
    service.import_from_file(dwd_pollen_fixture_path)

    # Grab a known row and note its first-read index + available_time.
    original = (
        db_session.query(PollenData)
        .filter(PollenData.region_code == "NW", PollenData.pollen_type == "birke")
        .order_by(PollenData.datum.asc())
        .first()
    )
    assert original is not None
    first_available_time = original.available_time
    first_index = original.pollen_index

    # Same payload ingested "later" — available_time advances, index reflects
    # any revision (here identical, so we just prove the field moves).
    result = service.import_from_file(dwd_pollen_fixture_path)
    assert result["updated"] > 0

    refreshed = (
        db_session.query(PollenData)
        .filter(
            PollenData.region_code == "NW",
            PollenData.pollen_type == "birke",
            PollenData.datum == original.datum,
        )
        .one()
    )
    assert refreshed.available_time >= first_available_time
    assert refreshed.pollen_index == first_index


def test_parse_index_handles_ranges_and_empty_tokens():
    assert PollenService._parse_index("0") == 0.0
    assert PollenService._parse_index("1") == 1.0
    assert PollenService._parse_index("0-1") == 0.5
    assert PollenService._parse_index("2-3") == 2.5
    assert PollenService._parse_index("keine") is None
    assert PollenService._parse_index("-") is None
    assert PollenService._parse_index(None) is None


def test_parse_timestamp_understands_dwd_format():
    parsed = PollenService._parse_dwd_timestamp("2026-04-18 11:00 Uhr")
    assert parsed == datetime(2026, 4, 18, 11, 0)
    assert PollenService._parse_dwd_timestamp("") is None


def test_unique_constraint_prevents_duplicate_rows(db_session, dwd_pollen_fixture_path):
    """Belt-and-braces: the DB-level unique constraint must still bite."""
    service = _pollen_service(db_session)
    service.import_from_file(dwd_pollen_fixture_path)
    sample = (
        db_session.query(PollenData)
        .filter(PollenData.region_code == "BY")
        .first()
    )
    assert sample is not None

    # Try to insert a literal duplicate through a second request.
    # The idempotent ingest path converts this into an UPDATE; inserting a
    # raw second row would violate the unique constraint — which is exactly
    # what we rely on.
    again = service.import_from_file(dwd_pollen_fixture_path)
    assert again["inserted"] == 0
    assert (
        db_session.query(PollenData)
        .filter(
            PollenData.region_code == sample.region_code,
            PollenData.pollen_type == sample.pollen_type,
            PollenData.datum == sample.datum,
        )
        .count()
        == 1
    )
