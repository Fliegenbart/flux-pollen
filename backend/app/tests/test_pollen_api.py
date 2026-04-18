"""End-to-end test for the pollen forecast API.

Seeds synthetic observations + weather, trains a ForecastService,
saves the artefact to a temporary registry directory, and calls the
API via FastAPI's TestClient. Ensures:

- ``/api/v1/pollen/current`` returns the latest ePIN observation
- ``/api/v1/pollen/forecast`` refuses to serve without a model (503)
  and returns a valid JSON with lower ≤ predicted ≤ upper after one
  is trained and saved
- ``/api/v1/pollen/forecast/regional`` returns the ranked list
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.database import PollenObservation, WeatherData
from app.services.ml.feature_engineering import (
    FeatureBuildConfig,
    assemble_training_frame,
    build_daily_panel,
)
from app.services.ml.forecast_service import ForecastService
from app.services.ml.model_registry import (
    ModelArtifact,
    build_metadata,
    save_artifact,
)


REGION = "BY"
POLLEN = "birke"
STATION = "DEMUNC"


def _seed(db_session, *, start: str, days: int) -> None:
    import numpy as np
    rng = np.random.default_rng(7)
    index = pd.date_range(start=start, periods=days, freq="D")
    doy = index.dayofyear.to_numpy().astype(float)
    season = 400.0 * np.exp(-((doy - 100) ** 2) / (2.0 * 18.0 ** 2))
    temp = 8.0 + 14.0 * np.sin(2 * 3.14159 * (doy - 90) / 365.25) + rng.normal(0, 2, size=days)
    humidity = np.clip(70 - (temp - 8), 30, 95)
    wind = np.clip(2.5 + rng.normal(0, 0.5, size=days), 0.5, 8)
    rain = np.maximum(rng.gamma(shape=1.5, scale=1.5, size=days) - 0.5, 0.0)
    noise = rng.normal(0, 8.0, size=days)
    concentration = np.maximum(season + noise, 0.0)

    now = datetime(2026, 1, 1)
    for datum, conc, t, h, w, r in zip(index, concentration, temp, humidity, wind, rain):
        db_session.add(
            PollenObservation(
                station_id=STATION,
                station_name="München",
                region_code=REGION,
                pollen_type=POLLEN,
                from_time=datum,
                to_time=datum + timedelta(hours=3),
                concentration=float(conc),
                algorithm="synthetic",
                source_network="ePIN",
                available_time=now,
                created_at=now,
            )
        )
        db_session.add(
            WeatherData(
                city="München",
                region_code=REGION,
                datum=datum,
                available_time=datum + timedelta(hours=20),
                temperatur=float(t),
                luftfeuchtigkeit=float(h),
                wind_geschwindigkeit=float(w),
                regen_mm=float(r),
                data_type="DAILY_OBSERVATION",
                created_at=now,
            )
        )
    db_session.commit()


@pytest.fixture
def ml_models_root(tmp_path, monkeypatch):
    from app.services.ml import model_registry

    root = tmp_path / "ml_models"
    monkeypatch.setattr(model_registry, "DEFAULT_ROOT", root)
    return root


@pytest.fixture
def app_client(db_session, monkeypatch):
    """FastAPI client where `get_db` yields the in-memory SQLite session."""
    from app.db.session import get_db

    app = create_app()

    def _override():
        yield db_session

    app.dependency_overrides[get_db] = _override
    return TestClient(app)


def test_current_endpoint_returns_latest_epin_observation(app_client, db_session):
    _seed(db_session, start="2024-10-01", days=180)
    response = app_client.get(
        "/api/v1/pollen/current",
        params={"region": REGION, "pollen_type": POLLEN},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pollen_type"] == POLLEN
    assert body["region_code"] == REGION
    assert body["source"] == "ePIN"
    assert body["concentration"] is not None


def test_forecast_endpoint_503_without_trained_model(app_client, db_session, ml_models_root):
    _seed(db_session, start="2024-10-01", days=180)
    response = app_client.get(
        "/api/v1/pollen/forecast",
        params={"region": REGION, "pollen_type": POLLEN, "horizon_days": 7},
    )
    assert response.status_code == 503


def test_forecast_endpoint_returns_bounded_forecast_after_training(
    app_client, db_session, ml_models_root
):
    _seed(db_session, start="2024-10-01", days=180)

    # Fit + save artefact straight to the patched root so the API loads it.
    start = datetime(2024, 10, 1)
    end = datetime(2025, 3, 29)  # aligns to the last seeded day
    config = FeatureBuildConfig(
        region_code=REGION, pollen_type=POLLEN, start_date=start, end_date=end
    )
    panel = build_daily_panel(db_session, config)
    X, y, index = assemble_training_frame(panel, horizon=7)
    service = ForecastService().fit(X, y)
    meta = build_metadata(
        service=service,
        feature_config=config,
        horizon_days=7,
        feature_columns=list(X.columns),
        train_n_samples=len(index),
        model_version="test-v0",
    )
    save_artifact(ModelArtifact(service=service, metadata=meta), root=ml_models_root)

    response = app_client.get(
        "/api/v1/pollen/forecast",
        params={"region": REGION, "pollen_type": POLLEN, "horizon_days": 7},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_version"] == "test-v0"
    forecast = body["forecast"]
    assert forecast["horizon_days"] == 7
    assert forecast["lower_bound"] <= forecast["predicted_concentration"] <= forecast["upper_bound"]
    assert forecast["lower_bound"] >= 0.0
    assert forecast["confidence_label"] in {"low", "medium", "high"}


def test_regional_ranking_returns_entries_for_trained_regions(
    app_client, db_session, ml_models_root
):
    _seed(db_session, start="2024-10-01", days=180)
    start = datetime(2024, 10, 1)
    end = datetime(2025, 3, 29)
    config = FeatureBuildConfig(
        region_code=REGION, pollen_type=POLLEN, start_date=start, end_date=end
    )
    panel = build_daily_panel(db_session, config)
    X, y, _ = assemble_training_frame(panel, horizon=7)
    service = ForecastService().fit(X, y)
    meta = build_metadata(
        service=service,
        feature_config=config,
        horizon_days=7,
        feature_columns=list(X.columns),
        train_n_samples=len(X),
        model_version="test-v0",
    )
    save_artifact(ModelArtifact(service=service, metadata=meta), root=ml_models_root)

    response = app_client.get(
        "/api/v1/pollen/forecast/regional",
        params={"pollen_type": POLLEN, "horizon_days": 7},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pollen_type"] == POLLEN
    assert body["horizon_days"] == 7
    assert len(body["entries"]) == 1
    assert body["entries"][0]["region_code"] == REGION
    assert body["entries"][0]["rank"] == 1


def test_forecast_rejects_unknown_horizon(app_client):
    response = app_client.get(
        "/api/v1/pollen/forecast",
        params={"region": REGION, "pollen_type": POLLEN, "horizon_days": 1},
    )
    assert response.status_code == 400


def test_forecast_rejects_unknown_region(app_client):
    response = app_client.get(
        "/api/v1/pollen/forecast",
        params={"region": "ZZ", "pollen_type": POLLEN, "horizon_days": 7},
    )
    assert response.status_code == 404
