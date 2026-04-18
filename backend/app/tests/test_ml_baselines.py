"""Unit tests for Persistence and Seasonal-Naive baselines."""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.baselines import (
    PersistenceBaseline,
    SeasonalNaiveBaseline,
)


def _daily_series(values: list[float], start: str = "2025-01-01") -> pd.Series:
    index = pd.date_range(start=start, periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float)


def test_persistence_predicts_last_observed_value_for_every_horizon():
    series = _daily_series([1.0, 2.0, 5.0, 4.0])
    model = PersistenceBaseline().fit(series)
    pred = model.predict(series, horizons=[1, 3, 7])

    assert pred.predicted.tolist() == [4.0, 4.0, 4.0]
    # Interval monotonically widens with horizon (sqrt(h) scaling).
    assert pred.upper[0] <= pred.upper[1] <= pred.upper[2]
    assert pred.lower[0] >= pred.lower[1] >= pred.lower[2]


def test_persistence_interval_is_non_degenerate_on_noisy_series():
    # Zero-mean noise around 50 — persistence's empirical residuals span
    # negative and positive values, so q10 < 0 < q90 and the interval
    # straddles the point prediction.
    rng = np.random.default_rng(0)
    values = 50.0 + rng.normal(0, 5.0, size=120)
    series = _daily_series(values.tolist())
    model = PersistenceBaseline().fit(series)
    pred = model.predict(series, horizons=[1])
    assert pred.lower[0] <= pred.predicted[0] <= pred.upper[0]
    assert pred.upper[0] - pred.lower[0] > 0.0


def test_persistence_interval_widens_with_horizon():
    rng = np.random.default_rng(1)
    values = 50.0 + rng.normal(0, 5.0, size=120)
    series = _daily_series(values.tolist())
    model = PersistenceBaseline().fit(series)
    pred = model.predict(series, horizons=[1, 7])
    width_h1 = pred.upper[0] - pred.lower[0]
    width_h7 = pred.upper[1] - pred.lower[1]
    assert width_h7 > width_h1


def test_seasonal_naive_reuses_last_year_same_day():
    values = list(range(1, 370))
    series = _daily_series(values, start="2024-01-01")
    model = SeasonalNaiveBaseline(season_length_days=365).fit(series)
    forecast_date = series.index[-1]  # 2025-01-03 because 369 days from 2024-01-01
    pred = model.predict(series, horizons=[1], forecast_date=forecast_date)
    # Target = forecast_date + 1d; lookup = target - 365d.
    target = forecast_date + pd.Timedelta(days=1)
    lookup = target - pd.Timedelta(days=365)
    assert pred.predicted[0] == pytest.approx(float(series.loc[lookup]))


def test_seasonal_naive_falls_back_gracefully_when_history_is_too_short():
    series = _daily_series([1.0, 2.0, 3.0])
    model = SeasonalNaiveBaseline(season_length_days=365).fit(series)
    pred = model.predict(
        series, horizons=[1], forecast_date=series.index[-1]
    )
    # No lookup candidate within 3-day tolerance → NaN.
    assert np.isnan(pred.predicted[0])


def test_persistence_zero_residuals_when_series_is_flat():
    series = _daily_series([7.0] * 50)
    model = PersistenceBaseline().fit(series)
    pred = model.predict(series, horizons=[1])
    assert pred.predicted[0] == 7.0
    # Zero residuals → zero-width interval.
    assert pred.lower[0] == pytest.approx(7.0)
    assert pred.upper[0] == pytest.approx(7.0)
