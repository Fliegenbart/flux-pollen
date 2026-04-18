"""Baselines for forecast evaluation.

Every sale starts with "better than what you'd get for free". Persistence
("tomorrow = today") and Seasonal-Naive ("next week = same week last
year") are the two free baselines customers will mentally benchmark
against. If the stacked ensemble does not beat these on WIS by a
comfortable margin, we have nothing to sell.

Both baselines here are intentionally plain — no smoothing, no
hyperparameters, no surprises. They exist to anchor the backtest
report, not to compete.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["PersistenceBaseline", "SeasonalNaiveBaseline", "BaselineForecast"]


@dataclass(frozen=True)
class BaselineForecast:
    """Point + empirical-quantile forecast from a baseline."""
    predicted: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


class PersistenceBaseline:
    """``ŷ(t + h) = y(t)`` for all horizons.

    Interval is built from the empirical absolute-residual quantiles of
    the baseline on the training window — an honest reflection of how
    uncertain the "no change" assumption is on this series.
    """

    def __init__(
        self,
        *,
        lower_quantile: float = 0.10,
        upper_quantile: float = 0.90,
    ) -> None:
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self._train_residuals: np.ndarray | None = None

    def fit(self, y_train: pd.Series) -> "PersistenceBaseline":
        """Record residuals ``y(t) − y(t − 1)`` so we can form an interval."""
        series = pd.Series(y_train).astype(float).dropna()
        if len(series) < 2:
            self._train_residuals = np.array([0.0])
            return self
        residuals = series.diff().dropna().to_numpy()
        self._train_residuals = residuals
        return self

    def predict(self, y_history: pd.Series, *, horizons: list[int]) -> BaselineForecast:
        last = float(pd.Series(y_history).astype(float).dropna().iloc[-1])
        horizons_arr = np.asarray(horizons, dtype=int)
        predicted = np.full(horizons_arr.shape, last, dtype=float)

        residuals = self._train_residuals if self._train_residuals is not None else np.array([0.0])
        q_low = float(np.quantile(residuals, self.lower_quantile))
        q_high = float(np.quantile(residuals, self.upper_quantile))
        # Interval widens with horizon: naive random-walk assumption says
        # variance scales linearly with h, so SD ∝ sqrt(h).
        scale = np.sqrt(horizons_arr.astype(float))
        lower = predicted + q_low * scale
        upper = predicted + q_high * scale
        return BaselineForecast(predicted=predicted, lower=lower, upper=upper)


class SeasonalNaiveBaseline:
    """``ŷ(t + h) = y(t + h − 365)`` (or configurable season length).

    Copies what happened the same day last year. For pollen this is a
    surprisingly hard baseline to beat outside the shoulder weeks, since
    the biology is strongly calendar-driven.
    """

    def __init__(
        self,
        *,
        season_length_days: int = 365,
        lower_quantile: float = 0.10,
        upper_quantile: float = 0.90,
    ) -> None:
        self.season_length_days = int(season_length_days)
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self._train_residuals: np.ndarray | None = None
        self._train_history: pd.Series | None = None

    def fit(self, y_train: pd.Series) -> "SeasonalNaiveBaseline":
        series = pd.Series(y_train).astype(float).dropna().sort_index()
        self._train_history = series
        if len(series) <= self.season_length_days:
            self._train_residuals = np.array([0.0])
            return self
        # Empirical residual = y(t) − y(t − season)
        shifted = series.shift(self.season_length_days)
        residuals = (series - shifted).dropna().to_numpy()
        self._train_residuals = residuals if len(residuals) else np.array([0.0])
        return self

    def predict(
        self,
        y_history: pd.Series,
        *,
        horizons: list[int],
        forecast_date: pd.Timestamp,
    ) -> BaselineForecast:
        series = pd.Series(y_history).astype(float).dropna().sort_index()
        # We need the same-day-last-year value for each horizon target.
        predicted = np.empty(len(horizons), dtype=float)
        for i, h in enumerate(horizons):
            target_date = pd.Timestamp(forecast_date) + pd.Timedelta(days=int(h))
            lookup_date = target_date - pd.Timedelta(days=self.season_length_days)
            predicted[i] = _nearest_value(series, lookup_date)

        residuals = self._train_residuals if self._train_residuals is not None else np.array([0.0])
        q_low = float(np.quantile(residuals, self.lower_quantile))
        q_high = float(np.quantile(residuals, self.upper_quantile))
        lower = predicted + q_low
        upper = predicted + q_high
        return BaselineForecast(predicted=predicted, lower=lower, upper=upper)


def _nearest_value(series: pd.Series, target: pd.Timestamp) -> float:
    """Value of ``series`` closest to ``target`` — within 3 days. Else NaN."""
    if series.empty:
        return float("nan")
    target = pd.Timestamp(target)
    idx = series.index
    if target in idx:
        return float(series.loc[target])
    deltas = pd.to_timedelta(idx - target)
    diffs_days = np.abs(deltas.total_seconds().to_numpy() / 86_400.0)
    position = int(np.argmin(diffs_days))
    if diffs_days[position] <= 3.0:
        return float(series.iloc[position])
    return float("nan")
