"""Walk-forward backtester for pollen forecasts.

The contract:

- Given a daily panel (see ``feature_engineering.build_daily_panel``)
  and a horizon h, we slide a training window forward one day at a time.
- At each step, train the forecast service on ``[0, t]``, predict
  ``t + h``, record metrics, compare to Persistence and Seasonal-Naive.
- Walk-forward ensures no look-ahead: the forecaster only ever sees
  history. Any violation here invalidates every downstream claim, so
  we keep the loop explicit and auditable.

Metrics returned mirror the canonical Hubverse panel: MAE, RMSE, pinball
at 0.1/0.5/0.9, WIS over the 80 % central interval, empirical coverage
of the 80 % interval, and the same metrics for each baseline. The
improvement numbers are relative WIS reductions vs. the best baseline.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.time import utc_now
from app.models.database import BacktestPoint, BacktestRun
from app.services.ml.baselines import PersistenceBaseline, SeasonalNaiveBaseline
from app.services.ml.feature_engineering import assemble_training_frame
from app.services.ml.forecast_service import ForecastOutput, ForecastService
from app.services.ml.metrics import (
    coverage,
    mae,
    pinball_loss,
    rmse,
    weighted_interval_score,
)


class ForecasterProtocol(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForecasterProtocol": ...
    def predict(self, X: pd.DataFrame) -> ForecastOutput: ...

logger = logging.getLogger(__name__)

__all__ = ["BacktestConfig", "BacktestResult", "WalkForwardBacktester"]


@dataclass
class BacktestConfig:
    horizon_days: int = 7
    min_train_days: int = 60
    step_days: int = 1
    lower_quantile: float = 0.10
    upper_quantile: float = 0.90
    seasonal_length_days: int = 365
    model_version: str = "pollencast-ridge-gbm-v0"
    forecaster_factory: Callable[["BacktestConfig"], ForecasterProtocol] | None = None

    def build_forecaster(self) -> ForecasterProtocol:
        if self.forecaster_factory is not None:
            return self.forecaster_factory(self)
        return ForecastService(
            lower_quantile=self.lower_quantile,
            upper_quantile=self.upper_quantile,
        )


@dataclass
class BacktestResult:
    run_id: str
    pollen_type: str
    region_code: str
    horizon_days: int
    n_folds: int
    metrics: dict[str, float]
    baseline_metrics: dict[str, dict[str, float]]
    improvement_vs_baselines: dict[str, float]
    points: list[dict[str, Any]] = field(default_factory=list)


class WalkForwardBacktester:
    def __init__(
        self,
        *,
        pollen_type: str,
        region_code: str,
        config: BacktestConfig | None = None,
    ) -> None:
        self.pollen_type = pollen_type
        self.region_code = region_code
        self.config = config or BacktestConfig()

    # ----------------------------------------------------------------- #
    # Main loop
    # ----------------------------------------------------------------- #

    def run(self, panel: pd.DataFrame) -> BacktestResult:
        horizon = self.config.horizon_days
        X, y, index = assemble_training_frame(panel, horizon=horizon)
        if len(index) < self.config.min_train_days + self.config.step_days:
            raise ValueError(
                f"Not enough data: {len(index)} usable rows, need >= "
                f"{self.config.min_train_days + self.config.step_days}."
            )

        records: list[dict[str, Any]] = []
        for fold_end in range(
            self.config.min_train_days,
            len(index) - 1,
            self.config.step_days,
        ):
            train_idx = index[:fold_end]
            test_idx = index[fold_end : fold_end + 1]
            if test_idx.empty:
                break

            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X.loc[test_idx]
            y_true_value = float(y.loc[test_idx].iloc[0])

            # Model forecast
            try:
                service = self.config.build_forecaster()
                service.fit(X_train, y_train)
                pred = service.predict(X_test)
                model_median = float(pred.predicted[0])
                model_lower = float(pred.lower[0])
                model_upper = float(pred.upper[0])
            except Exception as exc:
                logger.warning("Fold %s failed: %s", fold_end, exc)
                continue

            # Baselines operate on the raw target series up to "now" —
            # which in a direct h-step setup is ``train_idx[-1]``.
            history = panel.loc[:train_idx[-1], "y"].astype(float).dropna()
            persistence = PersistenceBaseline(
                lower_quantile=self.config.lower_quantile,
                upper_quantile=self.config.upper_quantile,
            ).fit(history)
            persistence_pred = persistence.predict(history, horizons=[horizon])
            seasonal = SeasonalNaiveBaseline(
                season_length_days=self.config.seasonal_length_days,
                lower_quantile=self.config.lower_quantile,
                upper_quantile=self.config.upper_quantile,
            ).fit(history)
            seasonal_pred = seasonal.predict(
                history, horizons=[horizon], forecast_date=train_idx[-1]
            )

            target_date = pd.Timestamp(train_idx[-1]) + pd.Timedelta(days=horizon)
            records.append(
                {
                    "fold": fold_end,
                    "forecast_date": pd.Timestamp(train_idx[-1]).to_pydatetime(),
                    "target_date": target_date.to_pydatetime(),
                    "y_true": y_true_value,
                    "model_median": model_median,
                    "model_lower": model_lower,
                    "model_upper": model_upper,
                    "persistence": float(persistence_pred.predicted[0]),
                    "persistence_lower": float(persistence_pred.lower[0]),
                    "persistence_upper": float(persistence_pred.upper[0]),
                    "seasonal": float(seasonal_pred.predicted[0])
                    if np.isfinite(seasonal_pred.predicted[0])
                    else np.nan,
                    "seasonal_lower": float(seasonal_pred.lower[0])
                    if np.isfinite(seasonal_pred.lower[0])
                    else np.nan,
                    "seasonal_upper": float(seasonal_pred.upper[0])
                    if np.isfinite(seasonal_pred.upper[0])
                    else np.nan,
                }
            )

        if not records:
            raise RuntimeError(
                "Walk-forward produced zero fold results — usable data too sparse "
                "after feature alignment. Check observation coverage and horizon."
            )

        frame = pd.DataFrame.from_records(records)
        metrics = self._summary_metrics(frame)
        baseline_metrics = self._baseline_metrics(frame)
        improvements = self._improvements(metrics, baseline_metrics)

        return BacktestResult(
            run_id=f"bt-{uuid.uuid4().hex[:10]}",
            pollen_type=self.pollen_type,
            region_code=self.region_code,
            horizon_days=horizon,
            n_folds=len(frame),
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            improvement_vs_baselines=improvements,
            points=frame.to_dict(orient="records"),
        )

    # ----------------------------------------------------------------- #
    # Summary helpers
    # ----------------------------------------------------------------- #

    def _summary_metrics(self, frame: pd.DataFrame) -> dict[str, float]:
        y = frame["y_true"].to_numpy()
        return {
            "mae": mae(y, frame["model_median"].to_numpy()),
            "rmse": rmse(y, frame["model_median"].to_numpy()),
            "pinball_q10": pinball_loss(
                y, frame["model_lower"].to_numpy(), self.config.lower_quantile
            ),
            "pinball_q50": pinball_loss(y, frame["model_median"].to_numpy(), 0.5),
            "pinball_q90": pinball_loss(
                y, frame["model_upper"].to_numpy(), self.config.upper_quantile
            ),
            "wis80": weighted_interval_score(
                y,
                frame["model_median"].to_numpy(),
                {
                    float(self.config.lower_quantile * 2.0): (
                        frame["model_lower"].to_numpy(),
                        frame["model_upper"].to_numpy(),
                    )
                },
            ),
            "coverage80": coverage(
                y,
                frame["model_lower"].to_numpy(),
                frame["model_upper"].to_numpy(),
            ),
        }

    def _baseline_metrics(self, frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        y = frame["y_true"].to_numpy()
        persistence_ok = frame["persistence"].notna()
        seasonal_ok = frame["seasonal"].notna()

        def _pack(preds: pd.Series, lower: pd.Series, upper: pd.Series, mask: pd.Series) -> dict[str, float]:
            if not mask.any():
                return {}
            y_m = y[mask.to_numpy()]
            p = preds[mask].to_numpy()
            lo = lower[mask].to_numpy()
            hi = upper[mask].to_numpy()
            return {
                "mae": mae(y_m, p),
                "rmse": rmse(y_m, p),
                "pinball_q50": pinball_loss(y_m, p, 0.5),
                "wis80": weighted_interval_score(
                    y_m, p, {0.2: (lo, hi)}
                ),
                "coverage80": coverage(y_m, lo, hi),
                "n": int(mask.sum()),
            }

        return {
            "persistence": _pack(
                frame["persistence"], frame["persistence_lower"], frame["persistence_upper"], persistence_ok
            ),
            "seasonal_naive": _pack(
                frame["seasonal"], frame["seasonal_lower"], frame["seasonal_upper"], seasonal_ok
            ),
        }

    def _improvements(
        self,
        metrics: dict[str, float],
        baseline_metrics: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Relative WIS improvement vs. each baseline (positive = model wins)."""
        out: dict[str, float] = {}
        model_wis = metrics.get("wis80")
        if model_wis is None or model_wis == 0:
            return out
        for name, bm in baseline_metrics.items():
            baseline_wis = bm.get("wis80")
            if baseline_wis is None or baseline_wis == 0 or np.isnan(baseline_wis):
                continue
            out[name] = float(1.0 - (model_wis / baseline_wis))
        return out


def persist_backtest_run(db: Session, result: BacktestResult, *, config: BacktestConfig) -> BacktestRun:
    run = BacktestRun(
        run_id=result.run_id,
        status="success",
        pollen_type=result.pollen_type,
        region_code=result.region_code,
        horizon_days=result.horizon_days,
        min_train_points=config.min_train_days,
        parameters={
            "min_train_days": config.min_train_days,
            "step_days": config.step_days,
            "seasonal_length_days": config.seasonal_length_days,
            "lower_quantile": config.lower_quantile,
            "upper_quantile": config.upper_quantile,
        },
        metrics=result.metrics,
        baseline_metrics=result.baseline_metrics,
        improvement_vs_baselines=result.improvement_vs_baselines,
        model_version=config.model_version,
        chart_points=result.n_folds,
        created_at=utc_now(),
    )
    db.add(run)
    db.flush()

    for point in result.points:
        db.add(
            BacktestPoint(
                run_id=run.run_id,
                date=point["target_date"],
                region_code=result.region_code,
                real_value=point["y_true"],
                predicted_value=point["model_median"],
                lower_bound=point["model_lower"],
                upper_bound=point["model_upper"],
                baseline_persistence=point["persistence"],
                baseline_seasonal=point.get("seasonal"),
                extra={
                    "forecast_date": point["forecast_date"].isoformat()
                    if hasattr(point["forecast_date"], "isoformat")
                    else str(point["forecast_date"]),
                    "fold": point.get("fold"),
                },
            )
        )
    db.commit()
    return run
