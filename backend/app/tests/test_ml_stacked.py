"""Tests for the stacked forecaster and its HW base.

The synthetic generator is the same one the single-stage backtester E2E
uses so the two results are directly comparable.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.services.ml.feature_engineering import (
    FeatureBuildConfig,
    assemble_training_frame,
    build_daily_panel,
)
from app.services.ml.holt_winters_forecaster import HoltWintersForecaster
from app.services.ml.stacked_forecast_service import StackedForecastService
from app.tests.test_ml_backtest_e2e import (
    POLLEN_TYPE,
    REGION_CODE,
    _seed_synthetic_panel,
    _synthetic_daily_series,
)


def _panel(db_session) -> pd.DataFrame:
    series = _synthetic_daily_series(start="2024-01-01", days=2 * 365)
    _seed_synthetic_panel(db_session, series)
    return build_daily_panel(
        db_session,
        FeatureBuildConfig(
            region_code=REGION_CODE,
            pollen_type=POLLEN_TYPE,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2025, 12, 31),
        ),
    )


def test_holt_winters_fits_and_predicts_non_negative_values():
    rng = np.random.default_rng(0)
    index = pd.date_range(start="2024-01-01", periods=400, freq="D")
    doy = index.dayofyear.to_numpy().astype(float)
    season = 200.0 * np.maximum(np.sin(2 * np.pi * (doy - 80) / 365.25), 0.0)
    y = pd.Series(season + rng.normal(0, 10, size=400), index=index)
    X = pd.DataFrame(index=index)

    hw = HoltWintersForecaster(horizon_days=7).fit(X, y)
    pred = hw.predict(pd.DataFrame(index=index[-5:]))
    assert pred.shape == (5,)
    assert np.all(pred >= 0.0)


def test_stacked_forecaster_round_trip_produces_bounded_forecast(db_session):
    panel = _panel(db_session)
    X, y, _ = assemble_training_frame(panel, horizon=7)

    service = StackedForecastService(horizon_days=7)
    service.fit(X, y)
    out = service.predict(X.iloc[-5:])

    assert out.predicted.shape == (5,)
    assert np.all(out.lower <= out.predicted)
    assert np.all(out.predicted <= out.upper)
    assert np.all(out.lower >= 0.0)


def test_stacked_forecaster_refuses_too_small_training(db_session):
    panel = _panel(db_session)
    X, y, _ = assemble_training_frame(panel, horizon=7)

    service = StackedForecastService(horizon_days=7)
    with pytest.raises(ValueError):
        service.fit(X.iloc[:40], y.iloc[:40])


def test_stacked_feature_columns_are_locked_at_fit_time(db_session):
    panel = _panel(db_session)
    X, y, _ = assemble_training_frame(panel, horizon=7)

    service = StackedForecastService(horizon_days=7)
    service.fit(X, y)

    # The meta learner remembered a specific column order — missing one
    # must produce a clear error, not a silently-wrong forecast. We
    # accept either the upstream sklearn ValueError (Ridge scaler sees
    # the wrong width first) or our own RuntimeError guard downstream.
    truncated = X.iloc[-3:].drop(columns=X.columns[0])
    with pytest.raises((RuntimeError, ValueError)):
        service.predict(truncated)
