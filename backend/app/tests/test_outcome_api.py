"""HTTP-level tests for the outcome endpoints."""

from __future__ import annotations

import io
import textwrap
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import create_app
from app.models.database import OutcomeObservation, PollenObservation


_VALID_CSV = textwrap.dedent(
    """\
    brand,product,region_code,week_start,metric,value
    hexal,lorano_5mg_20stk,BY,2024-04-01,sell_out_units,4321
    hexal,lorano_5mg_20stk,NW,2024-04-01,sell_out_units,7100
    hexal,lorano_5mg_20stk,BY,2024-04-01,sell_out_revenue_eur,38447
    """
)


@pytest.fixture
def app_client(db_session, monkeypatch):
    from app.services.ml import model_registry
    monkeypatch.setattr(model_registry, "DEFAULT_ROOT", model_registry.DEFAULT_ROOT / "unused")

    app = create_app()

    def _override():
        yield db_session

    app.dependency_overrides[get_db] = _override
    return TestClient(app)


def test_upload_endpoint_accepts_valid_csv(app_client, db_session):
    response = app_client.post(
        "/api/v1/outcome/upload",
        files={"file": ("demo.csv", _VALID_CSV, "text/csv")},
        params={"source_label": "pilot_hexal"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["brand"] == "hexal"
    assert body["rows_imported"] == 3
    assert set(body["regions_seen"]) == {"BY", "NW"}
    assert db_session.query(OutcomeObservation).count() == 3


def test_upload_endpoint_rejects_empty_file(app_client):
    response = app_client.post(
        "/api/v1/outcome/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400


def test_catalog_endpoint_lists_scope(app_client, db_session):
    app_client.post(
        "/api/v1/outcome/upload",
        files={"file": ("demo.csv", _VALID_CSV, "text/csv")},
    )
    response = app_client.get("/api/v1/outcome/catalog")
    assert response.status_code == 200
    body = response.json()
    entries = {(e["region_code"], e["metric"]) for e in body["entries"]}
    assert ("BY", "sell_out_units") in entries
    assert ("NW", "sell_out_units") in entries


def _seed_small_coupled_series(db_session):
    rng = np.random.default_rng(1)
    start = datetime(2024, 1, 1)
    now = datetime(2026, 1, 1)
    for day in range(200):
        date = start + timedelta(days=day)
        doy = date.timetuple().tm_yday
        pollen = max(0.0, 500.0 * np.exp(-((doy - 95) ** 2) / (2 * 15 ** 2)) + rng.normal(0, 10))
        db_session.add(
            PollenObservation(
                station_id="DEMUNC",
                station_name="München",
                region_code="BY",
                pollen_type="birke",
                from_time=date,
                to_time=date + timedelta(hours=3),
                concentration=float(pollen),
                algorithm="synthetic",
                source_network="ePIN",
                available_time=now,
                created_at=now,
            )
        )
    for w in range(28):
        ws = start + timedelta(days=7 * w)
        doy = (ws + timedelta(days=10)).timetuple().tm_yday  # 10-day forward coupling
        coupling = 500.0 * np.exp(-((doy - 95) ** 2) / (2 * 15 ** 2))
        value = 3000.0 + 1.5 * coupling + rng.normal(0, 30)
        db_session.add(
            OutcomeObservation(
                brand="hexal",
                product="lorano_5mg_20stk",
                region_code="BY",
                window_start=ws,
                window_end=ws + timedelta(days=7),
                metric_name="sell_out_units",
                metric_value=float(value),
                metric_unit="Packungen",
                source_label="test",
            )
        )
    db_session.commit()


def test_correlation_endpoint_returns_lag_and_lift(app_client, db_session):
    _seed_small_coupled_series(db_session)
    response = app_client.get(
        "/api/v1/outcome/correlation",
        params={
            "brand": "hexal",
            "product": "lorano_5mg_20stk",
            "region": "BY",
            "pollen_type": "birke",
            "metric": "sell_out_units",
            "max_lag_days": 21,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_weeks"] >= 15
    assert body["best_pearson"] > 0.7
    assert body["lift_high_vs_low_pct"] > 10.0
    # Lag curve is dense
    assert len(body["lag_curve"]) == 43  # -21..+21 inclusive


def test_correlation_endpoint_404_when_no_overlap(app_client):
    response = app_client.get(
        "/api/v1/outcome/correlation",
        params={
            "brand": "unknown",
            "product": "nothing",
            "region": "BY",
            "pollen_type": "birke",
        },
    )
    assert response.status_code == 404
