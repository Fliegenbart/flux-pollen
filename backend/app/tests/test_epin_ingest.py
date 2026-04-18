"""Integration tests for the ePIN ingest service.

Fixture is a real slice of the ePIN API response for 2024-03-25..2024-03-27
covering München and Hof × Betula/Poaceae/Alnus. That window captures the
front edge of a Birke bloom so the values are genuinely non-trivial.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from app.models.database import PollenObservation
from app.services.data_ingest.epin_service import EPINService


@pytest.fixture
def epin_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "epin_measurements_sample.json"


def test_import_from_fixture_persists_all_series(db_session, epin_fixture_path):
    service = EPINService(db_session)
    result = service.import_from_file(epin_fixture_path)

    assert result["success"] is True
    # 2 stations × 3 pollen species × 16 three-hour windows = 96 rows.
    stored = db_session.query(PollenObservation).count()
    assert stored == 96

    assert set(result["stations"]) == {"DEMUNC", "DEHOF"}
    assert set(result["pollen_types"]) == {"birke", "graeser", "erle"}
    # Every row must have an available_time (Point-in-Time anchor).
    assert (
        db_session.query(PollenObservation)
        .filter(PollenObservation.available_time.is_(None))
        .count()
        == 0
    )


def test_reingest_same_fixture_is_idempotent(db_session, epin_fixture_path):
    service = EPINService(db_session)
    service.import_from_file(epin_fixture_path)
    first = db_session.query(PollenObservation).count()

    result = service.import_from_file(epin_fixture_path)
    second = db_session.query(PollenObservation).count()

    assert first == second
    assert result["inserted"] == 0
    assert result["updated"] == first


def test_betula_peak_is_captured_with_correct_units(db_session, epin_fixture_path):
    """Concentrations in grains/m³ should survive the round-trip."""
    service = EPINService(db_session)
    service.import_from_file(epin_fixture_path)

    betula_peak = (
        db_session.query(PollenObservation)
        .filter(
            PollenObservation.station_id == "DEMUNC",
            PollenObservation.pollen_type == "birke",
        )
        .order_by(PollenObservation.concentration.desc())
        .first()
    )
    assert betula_peak is not None
    # The captured window sits on the front edge of the 2024 Birke bloom; by
    # the later peak (2024-04-07) DEMUNC read ~5900/m³. In this 48h slice the
    # peak is a few hundred — enough to prove the value survived the round-trip.
    assert betula_peak.concentration > 100.0
    assert betula_peak.region_code == "BY"
    assert betula_peak.station_name == "München"
    assert betula_peak.source_network == "ePIN"


def test_unknown_stations_and_species_are_silently_skipped(db_session):
    payload = {
        "from": 1700000000,
        "to": 1700100000,
        "measurements": [
            {
                "location": "DEMUNC",
                "polle": "Betula",
                "data": [
                    {"algorithm": "PomoAIv1.34.0", "from": 1700000000, "to": 1700010800, "value": 42.0},
                ],
            },
            {
                "location": "UNKNOWN",
                "polle": "Betula",
                "data": [
                    {"algorithm": "PomoAIv1.34.0", "from": 1700000000, "to": 1700010800, "value": 99.0},
                ],
            },
            {
                "location": "DEMUNC",
                "polle": "Abies",  # Not in our clinical subset — skipped.
                "data": [
                    {"algorithm": "PomoAIv1.34.0", "from": 1700000000, "to": 1700010800, "value": 7.0},
                ],
            },
        ],
    }

    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        fixture = fh.name

    service = EPINService(db_session)
    result = service.import_from_file(fixture)

    assert result["inserted"] == 1
    assert result["updated"] == 0
    assert (
        db_session.query(PollenObservation).count() == 1
    ), "Only the known DEMUNC/Betula row should be persisted."


def test_revised_value_updates_existing_row(db_session, epin_fixture_path):
    """When the API re-reports a window with a refined value, concentration updates."""
    service = EPINService(db_session)
    service.import_from_file(epin_fixture_path)

    sample = (
        db_session.query(PollenObservation)
        .filter(PollenObservation.pollen_type == "birke")
        .first()
    )
    assert sample is not None
    original_concentration = sample.concentration
    sample_station = sample.station_id
    sample_from = sample.from_time

    # Simulate a payload that revises the very first data point upward.
    import json
    payload = json.loads(epin_fixture_path.read_text())
    touched = False
    for series in payload["measurements"]:
        if series["location"] == sample_station and series["polle"] == "Betula":
            series["data"][0]["value"] = original_concentration + 1000.0
            touched = True
            break
    assert touched

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        mutated_fixture = fh.name

    service.import_from_file(mutated_fixture)

    refreshed = (
        db_session.query(PollenObservation)
        .filter(
            PollenObservation.station_id == sample_station,
            PollenObservation.pollen_type == "birke",
            PollenObservation.from_time == sample_from,
        )
        .one()
    )
    assert refreshed.concentration > original_concentration


def test_ingest_requires_known_bayern_stations(db_session):
    """Sanity: every resolved region_code on an ePIN row must be BY."""
    service = EPINService(db_session)
    service.import_from_file(
        Path(__file__).parent / "fixtures" / "epin_measurements_sample.json"
    )
    regions = {row.region_code for row in db_session.query(PollenObservation).all()}
    assert regions == {"BY"}
