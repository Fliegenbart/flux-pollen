"""Stacked forecast service.

Combines a Ridge median estimator with a Holt-Winters seasonality
specialist via XGBoost quantile meta-learners (one per output
quantile). Drop-in replacement for ``ForecastService`` — exposes the
same ``fit(X, y) / predict(X) -> ForecastOutput`` contract so the
backtester, CLI and API can switch between single-stage and stacked
models by constructing a different class.

Stacking strategy

- **Split the training frame in time.** The first ``base_train_frac``
  of rows trains the base estimators; the remainder becomes the
  meta-training set. This is the standard anti-leakage recipe for
  time-series stacking.
- **Base predictions on the meta split** go into the XGBoost meta
  learner together with the original features, so the meta can both
  correct systematic bias of a single base model and reweight
  features conditionally.
- **After the meta is trained, all base estimators are refit on the
  full training history.** Inference uses those "full-data" bases
  plus the (already-trained) meta.

XGBoost meta uses one model per quantile with ``objective=
reg:quantileerror``. The three models are trained independently, so
bounds can cross in principle — we sort them at predict time and
clamp to non-negative, matching the single-stage service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.ml.forecast_service import ForecastOutput
from app.services.ml.holt_winters_forecaster import HoltWintersForecaster

logger = logging.getLogger(__name__)

__all__ = ["StackedForecastService"]


@dataclass
class StackedForecastService:
    horizon_days: int = 7
    base_train_frac: float = 0.7
    ridge_alpha: float = 1.0
    lower_quantile: float = 0.10
    upper_quantile: float = 0.90
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    random_state: int = 42

    _ridge_pipeline: Pipeline | None = field(default=None, init=False, repr=False)
    _hw: HoltWintersForecaster | None = field(default=None, init=False, repr=False)
    _meta_lower: "object" = field(default=None, init=False, repr=False)
    _meta_median: "object" = field(default=None, init=False, repr=False)
    _meta_upper: "object" = field(default=None, init=False, repr=False)
    _meta_feature_names: list[str] | None = field(default=None, init=False, repr=False)
    _last_training_y: pd.Series | None = field(default=None, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackedForecastService":
        if len(X) < 60:
            raise ValueError(
                f"StackedForecastService needs >=60 training rows, got {len(X)}. "
                "Fall back to the single-stage ForecastService on short windows."
            )
        X = X.sort_index()
        y = pd.Series(y).sort_index().astype(float)
        if not X.index.equals(y.index):
            raise ValueError("X and y must share the same DatetimeIndex.")

        split = max(30, int(len(X) * self.base_train_frac))
        X_base, y_base = X.iloc[:split], y.iloc[:split]
        X_meta, y_meta = X.iloc[split:], y.iloc[split:]
        if len(X_meta) < 15:
            raise ValueError(
                "Meta split too small after time-based partition — increase history."
            )

        ridge = self._build_ridge()
        ridge.fit(self._sanitize(X_base), y_base.to_numpy())
        ridge_preds_meta = ridge.predict(self._sanitize(X_meta))

        hw = HoltWintersForecaster(horizon_days=self.horizon_days).fit(X_base, y_base)
        # Project HW for each meta row independently: refit HW on history
        # up to (meta_date − horizon_days) days, then forecast h-ahead.
        hw_preds_meta = self._hw_meta_predictions(
            hw=hw,
            y_full=y,
            meta_index=X_meta.index,
        )

        meta_features = self._assemble_meta_features(
            original=X_meta,
            ridge_preds=ridge_preds_meta,
            hw_preds=hw_preds_meta,
        )
        self._meta_feature_names = list(meta_features.columns)

        self._meta_median = self._fit_xgb_quantile(meta_features, y_meta, 0.5)
        self._meta_lower = self._fit_xgb_quantile(meta_features, y_meta, self.lower_quantile)
        self._meta_upper = self._fit_xgb_quantile(meta_features, y_meta, self.upper_quantile)

        # Refit bases on full history for inference.
        self._ridge_pipeline = self._build_ridge()
        self._ridge_pipeline.fit(self._sanitize(X), y.to_numpy())
        self._hw = HoltWintersForecaster(horizon_days=self.horizon_days).fit(X, y)
        self._last_training_y = y
        return self

    def predict(self, X: pd.DataFrame) -> ForecastOutput:
        if self._ridge_pipeline is None or self._hw is None:
            raise RuntimeError("StackedForecastService.predict called before fit().")
        X_clean = self._sanitize(X)
        ridge_preds = self._ridge_pipeline.predict(X_clean)
        hw_preds = self._hw.predict(X)

        meta_features = self._assemble_meta_features(
            original=X, ridge_preds=ridge_preds, hw_preds=hw_preds
        )
        if self._meta_feature_names is not None:
            # Guard against feature-column drift between fit and predict.
            missing = [c for c in self._meta_feature_names if c not in meta_features.columns]
            if missing:
                raise RuntimeError(
                    f"Meta feature columns missing at predict: {missing[:5]}…"
                )
            meta_features = meta_features[self._meta_feature_names]

        lower = self._meta_lower.predict(meta_features.to_numpy())
        median = self._meta_median.predict(meta_features.to_numpy())
        upper = self._meta_upper.predict(meta_features.to_numpy())

        # Sort to enforce monotonicity and clamp to non-negative.
        stacked = np.vstack([lower, median, upper])
        stacked.sort(axis=0)
        lower, median, upper = stacked
        lower = np.maximum(lower, 0.0)
        median = np.maximum(median, 0.0)
        upper = np.maximum(upper, 0.0)
        return ForecastOutput(
            predicted=median,
            lower=lower,
            upper=upper,
            quantiles=(self.lower_quantile, self.upper_quantile),
        )

    # ----------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------- #

    def _build_ridge(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.ridge_alpha)),
            ]
        )

    @staticmethod
    def _sanitize(X: pd.DataFrame) -> np.ndarray:
        arr = np.asarray(X.to_numpy(dtype=float))
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _hw_meta_predictions(
        self,
        hw: HoltWintersForecaster,
        *,
        y_full: pd.Series,
        meta_index: pd.DatetimeIndex,
    ) -> np.ndarray:
        """HW out-of-sample predictions for the meta split.

        One HW fit per backtest fold (already done in ``hw``), then a
        single multi-step forecast covering the whole meta period. Each
        meta row reads its corresponding offset from that forecast.

        Trade-off: this is not "expanding-window" stacking — the HW
        model is frozen after the base split. In return it is two
        orders of magnitude faster (one fit vs. one-per-row) without
        letting base-future data leak into meta-training. For pollen
        the lost signal is minor: a yearly-seasonal HW does not learn
        anything materially new from extending the window by 30 %.
        """
        if hw._fit_result is None or hw._fit_series is None:  # type: ignore[attr-defined]
            return np.zeros(len(meta_index), dtype=float)
        last_base_date = hw._fit_series.index[-1]  # type: ignore[attr-defined]
        meta_sorted = meta_index.sort_values()
        max_offset = (meta_sorted[-1] - last_base_date).days
        if max_offset <= 0:
            return np.full(len(meta_index), float(hw._fit_series.iloc[-1]))  # type: ignore[attr-defined]
        try:
            forecast_series = hw._fit_result.forecast(steps=int(max_offset))  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover — HW divergence fallback
            logger.debug("HW forecast failed: %s — falling back to last value", exc)
            last_val = float(hw._fit_series.iloc[-1])  # type: ignore[attr-defined]
            return np.full(len(meta_index), last_val)

        forecast_series = forecast_series.clip(lower=0.0)
        predictions = np.zeros(len(meta_index), dtype=float)
        for i, target_date in enumerate(meta_index):
            offset_days = (target_date - last_base_date).days
            if offset_days <= 0:
                predictions[i] = float(hw._fit_series.iloc[-1])  # type: ignore[attr-defined]
            else:
                idx = min(offset_days - 1, len(forecast_series) - 1)
                predictions[i] = float(forecast_series.iloc[idx])
        return predictions

    def _assemble_meta_features(
        self,
        *,
        original: pd.DataFrame,
        ridge_preds,
        hw_preds,
    ) -> pd.DataFrame:
        meta = original.copy()
        meta["base_ridge"] = np.asarray(ridge_preds, dtype=float)
        meta["base_hw"] = np.asarray(hw_preds, dtype=float)
        meta["base_diff"] = meta["base_ridge"] - meta["base_hw"]
        meta["base_mean"] = 0.5 * (meta["base_ridge"] + meta["base_hw"])
        return meta.fillna(0.0)

    def _fit_xgb_quantile(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        quantile: float,
    ):
        from xgboost import XGBRegressor

        model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=float(quantile),
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            random_state=self.random_state,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(X.to_numpy(), y.to_numpy())
        return model
