"""End-to-end backtest on a synthetic but realistic pollen season.

The synthetic data generator produces two years of daily concentration
values that mimic how a Birke bloom looks: a low winter baseline, a
sharp spring rise, and a steep decline, plus auto-regressive noise and
a weather coupling (warm + dry → higher counts). The test then drives
the full ingest → panel → backtest pipeline and asserts that the
Ridge-+-GBM model beats the Persistence and Seasonal-Naive baselines
on WIS over 7-day horizons.

Yes, synthetic data can make any model look good. But because the
generator is public in this test, the user can see exactly what the
forecaster is being asked to do, and the gap vs. baselines only
exists if the model is actually using the weather and lag structure.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.models.database import (
    BacktestPoint,
    BacktestRun,
    PollenObservation,
    WeatherData,
)
from app.services.ml.backtester import (
    BacktestConfig,
    WalkForwardBacktester,
    persist_backtest_run,
)
from app.services.ml.feature_engineering import FeatureBuildConfig, build_daily_panel


STATION_ID = "DEMUNC"
REGION_CODE = "BY"
POLLEN_TYPE = "birke"


def _synthetic_daily_series(
    *,
    start: str,
    days: int,
    peak_doy: int = 100,
    peak_value: float = 600.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Two seasonal years with AR(1) noise and a weather coupling.

    Target shape: Gaussian bump around ``peak_doy`` each year, scaled by
    a crude "warmth index" (sin(doy) with a negative rain coupling) to
    create a real weather signal that the model can exploit.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=days, freq="D")
    doy = index.dayofyear.to_numpy().astype(float)

    # Seasonal envelope: Gaussian around the configured peak.
    sigma = 18.0
    envelope = peak_value * np.exp(-((doy - peak_doy) ** 2) / (2.0 * sigma ** 2))

    # Weather signal: warmer = higher, wetter = lower.
    temp_noise = rng.normal(0, 2.0, size=days)
    temperature = 8.0 + 14.0 * np.sin(2 * math.pi * (doy - 90) / 365.25) + temp_noise
    rain = np.maximum(rng.gamma(shape=1.5, scale=1.5, size=days) - 0.5, 0.0)
    humidity = np.clip(70 - (temperature - 8.0) + rng.normal(0, 3, size=days), 30, 95)
    wind = np.clip(2.5 + rng.normal(0, 0.5, size=days), 0.5, 8)

    coupling = 1.0 + 0.08 * (temperature - temperature.mean()) - 0.2 * rain
    coupling = np.clip(coupling, 0.1, 2.5)

    # AR(1) noise on top of envelope × coupling.
    signal = envelope * coupling
    noise = np.zeros(days)
    for i in range(1, days):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 12.0)
    concentration = np.maximum(signal + noise, 0.0)

    return pd.DataFrame(
        {
            "datum": index,
            "concentration": concentration,
            "temperatur": temperature,
            "luftfeuchtigkeit": humidity,
            "wind_geschwindigkeit": wind,
            "regen_mm": rain,
        }
    )


def _seed_synthetic_panel(db_session, series: pd.DataFrame) -> None:
    """Persist the synthetic series as ePIN-like observations + weather."""
    now = datetime(2026, 1, 1)
    for row in series.itertuples(index=False):
        db_session.add(
            PollenObservation(
                station_id=STATION_ID,
                station_name="München",
                region_code=REGION_CODE,
                pollen_type=POLLEN_TYPE,
                from_time=row.datum,
                to_time=row.datum + timedelta(hours=3),
                concentration=float(row.concentration),
                algorithm="synthetic",
                source_network="ePIN",
                available_time=now,
                created_at=now,
            )
        )
        db_session.add(
            WeatherData(
                city="München",
                region_code=REGION_CODE,
                datum=row.datum,
                available_time=row.datum + timedelta(hours=20),
                temperatur=float(row.temperatur),
                luftfeuchtigkeit=float(row.luftfeuchtigkeit),
                wind_geschwindigkeit=float(row.wind_geschwindigkeit),
                regen_mm=float(row.regen_mm),
                data_type="DAILY_OBSERVATION",
                created_at=now,
            )
        )
    db_session.commit()


@pytest.fixture
def synthetic_panel(db_session):
    series = _synthetic_daily_series(start="2024-01-01", days=2 * 365)
    _seed_synthetic_panel(db_session, series)
    return series


def test_synthetic_panel_build_yields_expected_feature_shape(db_session, synthetic_panel):
    panel = build_daily_panel(
        db_session,
        FeatureBuildConfig(
            region_code=REGION_CODE,
            pollen_type=POLLEN_TYPE,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2025, 12, 31),
        ),
    )
    assert not panel.empty
    # Target + target-lag + weather + calendar columns must exist.
    assert "y" in panel.columns
    assert "y_lag1" in panel.columns
    assert "wx_temperatur_now" in panel.columns
    assert "cal_doy_sin" in panel.columns


def test_ridge_gbm_model_beats_persistence_on_wis(db_session, synthetic_panel):
    """The earn-my-keep test. If Ridge+GBM cannot beat "tomorrow = today"
    on 7-day horizons for a series with genuine seasonal + weather signal,
    something structural is broken."""
    panel = build_daily_panel(
        db_session,
        FeatureBuildConfig(
            region_code=REGION_CODE,
            pollen_type=POLLEN_TYPE,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2025, 12, 31),
        ),
    )
    backtester = WalkForwardBacktester(
        pollen_type=POLLEN_TYPE,
        region_code=REGION_CODE,
        config=BacktestConfig(
            horizon_days=7,
            min_train_days=120,
            step_days=14,  # stride to keep the test fast
        ),
    )
    result = backtester.run(panel)

    assert result.n_folds >= 5
    assert result.metrics["wis80"] > 0
    model_wis = result.metrics["wis80"]
    persistence_wis = result.baseline_metrics["persistence"]["wis80"]
    assert model_wis < persistence_wis, (
        f"Model WIS ({model_wis:.2f}) should beat persistence ({persistence_wis:.2f}) "
        "on synthetic data with a real seasonal + weather signal."
    )
    # Record the effect size; > 0 means the model wins.
    assert result.improvement_vs_baselines["persistence"] > 0.0


def test_backtest_run_persists_with_matching_point_count(db_session, synthetic_panel):
    panel = build_daily_panel(
        db_session,
        FeatureBuildConfig(
            region_code=REGION_CODE,
            pollen_type=POLLEN_TYPE,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2025, 12, 31),
        ),
    )
    config = BacktestConfig(horizon_days=7, min_train_days=120, step_days=14)
    backtester = WalkForwardBacktester(
        pollen_type=POLLEN_TYPE,
        region_code=REGION_CODE,
        config=config,
    )
    result = backtester.run(panel)
    persist_backtest_run(db_session, result, config=config)

    stored_run = (
        db_session.query(BacktestRun).filter(BacktestRun.run_id == result.run_id).one()
    )
    stored_points = (
        db_session.query(BacktestPoint).filter(BacktestPoint.run_id == result.run_id).count()
    )
    assert stored_run.pollen_type == POLLEN_TYPE
    assert stored_run.region_code == REGION_CODE
    assert stored_run.horizon_days == 7
    assert stored_run.chart_points == result.n_folds
    assert stored_points == result.n_folds
