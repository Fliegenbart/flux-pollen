"""Tests for the Split-Conformal calibration wrapper."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.services.ml.conformal_calibrator import ConformalCalibratedForecaster
from app.services.ml.feature_engineering import (
    FeatureBuildConfig,
    assemble_training_frame,
    build_daily_panel,
)
from app.services.ml.forecast_service import ForecastService
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


def test_conformal_widens_intervals_relative_to_base(db_session):
    panel = _panel(db_session)
    X, y, _ = assemble_training_frame(panel, horizon=7)

    # Base: intentionally sharp — trained only on the first 60% so its
    # residuals will be sizeable on the calibration tail.
    base = ForecastService()
    calibrator = ConformalCalibratedForecaster(base=base, target_coverage=0.80)
    calibrator.fit(X, y)

    # Refit a plain service on the same data for a fair width comparison.
    fresh_base = ForecastService().fit(X, y)

    sample = X.iloc[-20:]
    base_pred = fresh_base.predict(sample)
    cal_pred = calibrator.predict(sample)

    base_widths = base_pred.upper - base_pred.lower
    cal_widths = cal_pred.upper - cal_pred.lower
    # Conformal never shrinks; almost always widens.
    assert np.all(cal_widths + 1e-9 >= base_widths)


def test_conformal_records_calibration_summary(db_session):
    panel = _panel(db_session)
    X, y, _ = assemble_training_frame(panel, horizon=7)

    calibrator = ConformalCalibratedForecaster(base=ForecastService(), target_coverage=0.80)
    calibrator.fit(X, y)
    summary = calibrator.calibration_summary

    assert summary["target_coverage"] == 0.80
    assert summary["calibration_n"] > 0
    assert summary["width_adjustment"] >= 0.0
    assert 0.0 <= summary["calibration_raw_coverage"] <= 1.0


def test_conformal_predict_output_shape_matches_input(db_session):
    panel = _panel(db_session)
    X, y, _ = assemble_training_frame(panel, horizon=7)

    calibrator = ConformalCalibratedForecaster(base=ForecastService())
    calibrator.fit(X, y)
    pred = calibrator.predict(X.iloc[-5:])

    assert pred.predicted.shape == (5,)
    assert pred.lower.shape == (5,)
    assert pred.upper.shape == (5,)
    assert np.all(pred.lower >= 0.0)
    assert np.all(pred.lower <= pred.predicted)
    assert np.all(pred.predicted <= pred.upper)


def test_conformal_rejects_invalid_target_coverage(db_session):
    panel = _panel(db_session)
    X, y, _ = assemble_training_frame(panel, horizon=7)

    calibrator = ConformalCalibratedForecaster(
        base=ForecastService(), target_coverage=1.5
    )
    with pytest.raises(ValueError):
        calibrator.fit(X, y)


def test_conformal_rejects_insufficient_training_rows(db_session):
    _seed_synthetic_panel(
        db_session,
        _synthetic_daily_series(start="2024-01-01", days=80),
    )
    panel = build_daily_panel(
        db_session,
        FeatureBuildConfig(
            region_code=REGION_CODE,
            pollen_type=POLLEN_TYPE,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 1),
        ),
    )
    X, y, _ = assemble_training_frame(panel, horizon=7)
    # Calibration split wants >= 30 rows; this short sample cannot provide that.
    calibrator = ConformalCalibratedForecaster(
        base=ForecastService(), calibration_frac=0.2
    )
    with pytest.raises((ValueError, RuntimeError)):
        calibrator.fit(X.iloc[:35], y.iloc[:35])
