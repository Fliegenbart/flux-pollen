"""Forecast orchestrator for a single (region, pollen_type, horizon).

This is the minimum useful forecaster for Milestone 1: a Ridge regressor
on the full feature panel, with interval bounds derived from in-sample
quantile-regression (pinball loss at q ∈ {0.1, 0.9}).

The interface is deliberately small — ``fit(X, y)`` / ``predict(X)`` —
so we can slot in Holt-Winters, Prophet, and an XGBoost stacking meta-
learner in Phase 4b without touching the backtester or the CLI.

Notes on the Ridge choice:

- Pollen time series are strongly autocorrelated; with lag features the
  target becomes a smooth function of history + weather, which Ridge
  handles gracefully and without hyperparameter drama.
- Quantile regression via pinball-loss GBM (scikit-learn
  ``GradientBoostingRegressor(loss="quantile")``) gives us honest lower/
  upper bands without imposing Gaussianity. A parametric bootstrap on
  Ridge residuals would work too, but pinball GBM is the standard the
  RKI and CDC forecasting hubs use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["ForecastService", "ForecastOutput"]


@dataclass(frozen=True)
class ForecastOutput:
    """Median + interval bounds returned by ``ForecastService.predict``."""
    predicted: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    quantiles: tuple[float, float] = (0.10, 0.90)


@dataclass
class ForecastService:
    """Stacking-ready forecaster. Phase 4a ships the single-estimator version."""

    alpha: float = 1.0
    lower_quantile: float = 0.10
    upper_quantile: float = 0.90
    gbm_n_estimators: int = 150
    gbm_max_depth: int = 3
    gbm_learning_rate: float = 0.05
    random_state: int = 42
    _median_model: Pipeline | None = field(default=None, init=False, repr=False)
    _lower_model: GradientBoostingRegressor | None = field(default=None, init=False, repr=False)
    _upper_model: GradientBoostingRegressor | None = field(default=None, init=False, repr=False)
    _y_scale: float = field(default=1.0, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForecastService":
        X_arr, y_arr = self._to_arrays(X, y)
        if X_arr.shape[0] < 10:
            raise ValueError(
                f"ForecastService needs >=10 training points, got {X_arr.shape[0]}. "
                "Extend the training window or use a baseline."
            )
        self._y_scale = float(np.std(y_arr) or 1.0)

        self._median_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self._median_model.fit(X_arr, y_arr)

        self._lower_model = self._fit_quantile(X_arr, y_arr, self.lower_quantile)
        self._upper_model = self._fit_quantile(X_arr, y_arr, self.upper_quantile)
        return self

    def _fit_quantile(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quantile: float,
    ) -> GradientBoostingRegressor:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=float(quantile),
            n_estimators=self.gbm_n_estimators,
            max_depth=self.gbm_max_depth,
            learning_rate=self.gbm_learning_rate,
            random_state=self.random_state,
        )
        model.fit(X, y)
        return model

    def predict(self, X: pd.DataFrame) -> ForecastOutput:
        if self._median_model is None or self._lower_model is None or self._upper_model is None:
            raise RuntimeError("ForecastService.predict called before fit().")
        X_arr = self._coerce_features(X)

        predicted = self._median_model.predict(X_arr)
        lower = self._lower_model.predict(X_arr)
        upper = self._upper_model.predict(X_arr)

        # Enforce monotonicity lower <= predicted <= upper. In-sample the
        # quantile GBMs can cross each other by a small margin; we pin
        # them to the median if that happens.
        lower = np.minimum(lower, predicted)
        upper = np.maximum(upper, predicted)
        # Concentrations are non-negative.
        lower = np.maximum(lower, 0.0)
        predicted = np.maximum(predicted, 0.0)
        upper = np.maximum(upper, 0.0)
        return ForecastOutput(
            predicted=predicted,
            lower=lower,
            upper=upper,
            quantiles=(self.lower_quantile, self.upper_quantile),
        )

    # ----------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------- #

    @staticmethod
    def _to_arrays(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X.to_numpy(dtype=float))
        y_arr = np.asarray(pd.Series(y).astype(float).to_numpy())
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X_arr.shape[0]} vs {y_arr.shape[0]}"
            )
        # Replace NaNs in features with 0 as a last line of defence; the
        # feature pipeline is supposed to drop them upstream.
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return X_arr, y_arr

    @staticmethod
    def _coerce_features(X: pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X.to_numpy(dtype=float))
        return np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
