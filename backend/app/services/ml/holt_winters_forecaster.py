"""Holt-Winters base estimator for the stacked forecaster.

Plays the role of the seasonality specialist in the ensemble. Where
Ridge learns "given today's features, what is the daily mean in h
days", Holt-Winters learns "given the last N observations of the
target alone, extrapolate the level + trend + yearly seasonal".

Interface deliberately mirrors ``ForecastService.predict`` so the
stacking layer treats every base estimator uniformly — but only the
``predicted`` ndarray is used: HW's built-in prediction intervals are
Gaussian and unreliable on pollen, so we let the XGBoost quantile
stacker carry the uncertainty calibration downstream.

Design notes

- The input ``X`` DataFrame is carried through only for its
  DatetimeIndex; HW itself never consumes its columns.
- ``fit_series`` caches the target series and refits at predict time
  on the full history. That matches how the stacker is expected to
  call it: fit on the base-train split, predict on the meta-train
  split, then refit on full history and predict on the inference row.
- Seasonal mode is additive with ``seasonal_periods=365``; multi-
  plicative fails catastrophically on pollen because there are long
  zero runs in the off-season.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["HoltWintersForecaster"]


@dataclass
class HoltWintersForecaster:
    seasonal_periods: int = 365
    trend: str | None = "add"
    seasonal: str | None = "add"
    use_boxcox: bool = False
    horizon_days: int = 7
    _fit_series: pd.Series | None = field(default=None, init=False, repr=False)
    _fitted_values: pd.Series | None = field(default=None, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HoltWintersForecaster":
        """Fit on a target series indexed by DatetimeIndex.

        ``X`` is accepted for interface parity but its columns are
        ignored. ``y`` must share the same DatetimeIndex as ``X``.
        """
        series = pd.Series(y).astype(float)
        # statsmodels' heuristic initializer needs at least two full
        # seasonal cycles; below that we fit trend-only, still cheap.
        if len(series) < 2 * self.seasonal_periods + 10:
            seasonal = None
            seasonal_periods = None
        else:
            seasonal = self.seasonal
            seasonal_periods = self.seasonal_periods

        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
                use_boxcox=self.use_boxcox,
            )
            self._fit_result = self._model.fit(optimized=True, remove_bias=True)
            self._fitted_values = self._fit_result.fittedvalues
        self._fit_series = series
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return an h-step-ahead point forecast for each row in X.

        ``len(X)`` is the number of forecast rows to emit; each row's
        timestamp defines the *target* date, and HW projects
        ``horizon_days`` ahead of the last fit observation.
        """
        if self._fit_series is None or self._fit_result is None:
            raise RuntimeError("HoltWintersForecaster.predict called before fit().")

        # Note: we check row count, not ``.empty`` — a DataFrame whose
        # only job is to carry a DatetimeIndex has zero columns but is
        # still a valid input for a point forecast.
        if len(X) == 0:
            return np.empty(0, dtype=float)

        # Forecast far enough to cover the furthest requested target.
        max_steps = max(1, int(self.horizon_days))
        forecast_vals = self._fit_result.forecast(steps=max_steps)
        # Repeat the last horizon forecast across any rows that ask for it.
        predicted = np.full(len(X), float(forecast_vals.iloc[-1]))
        predicted = np.maximum(predicted, 0.0)
        return predicted

    def in_sample_predict(self) -> pd.Series:
        """In-sample fitted values — used by the stacker for meta-training."""
        if self._fitted_values is None:
            raise RuntimeError("HoltWintersForecaster.in_sample_predict before fit().")
        return self._fitted_values.astype(float).fillna(0.0).clip(lower=0.0)
