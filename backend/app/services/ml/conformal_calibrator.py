"""Split-Conformal calibration wrapper.

Wraps any object that satisfies the ``ForecasterProtocol`` (both the
single-stage ``ForecastService`` and the stacked version qualify) and
post-processes its prediction intervals so empirical coverage matches
a target level — typically the 80 % nominal band.

Background. The stacking layer produces sharper intervals than the
single-stage quantile GBM, but on real ePIN data the coverage drops
from ~0.90 (overwide) to ~0.60 (too narrow). WIS rewards that
sharpness as long as it remains correct, but honest forecasts must
honour their coverage label — a "80 % band" with 60 % empirical
coverage is misleading.

Split-Conformal Prediction (SCP, Vovk et al. 2005; modern survey:
Angelopoulos & Bates 2023) solves exactly that:

    score_i = max(lower_i − y_i, y_i − upper_i, 0)
    width = Quantile_{target} { score_1, ..., score_n } × finite-sample correction
    calibrated_lower  = predicted_lower  − width
    calibrated_upper  = predicted_upper  + width

Under exchangeability the calibrated interval has marginal coverage
≥ target. For time series strict exchangeability is violated, but
splitting into base-train / conformity-calibration by time still works
in practice and is the standard choice in the forecast-hub literature.

Design notes

- ``fit`` reserves the last ``calibration_frac`` of training rows for
  conformity scoring, fits the base on the rest, records ``width``,
  then refits the base on *all* data so inference uses every sample.
- ``width`` can only grow intervals. If the user's base already
  over-covers, SCP leaves it as-is (score distribution ≈ zero) — the
  intervals do not shrink. For shrinking you would use Conformalized
  Quantile Regression (CQR), which would be Phase 5 work.
- The wrapper shares the ``ForecastOutput`` contract so downstream
  code (API endpoint, backtester) stays unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from app.services.ml.forecast_service import ForecastOutput

logger = logging.getLogger(__name__)

__all__ = ["ConformalCalibratedForecaster"]


class _HasFitPredict(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series): ...
    def predict(self, X: pd.DataFrame) -> ForecastOutput: ...


@dataclass
class ConformalCalibratedForecaster:
    base: _HasFitPredict
    target_coverage: float = 0.80
    calibration_frac: float = 0.2
    _width: float = field(default=0.0, init=False, repr=False)
    _calibration_n: int = field(default=0, init=False, repr=False)
    _calibration_raw_coverage: float = field(default=0.0, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConformalCalibratedForecaster":
        if not 0.0 < self.calibration_frac < 0.5:
            raise ValueError(
                f"calibration_frac must be in (0, 0.5), got {self.calibration_frac}"
            )
        if not 0.0 < self.target_coverage < 1.0:
            raise ValueError(
                f"target_coverage must be in (0, 1), got {self.target_coverage}"
            )

        X = X.sort_index()
        y = pd.Series(y).sort_index().astype(float)
        n = len(X)
        cal_size = max(30, int(n * self.calibration_frac))
        if cal_size >= n:
            raise ValueError(
                "Not enough training rows for a conformal split. "
                "Need > 30 rows for calibration."
            )
        split = n - cal_size
        X_base, y_base = X.iloc[:split], y.iloc[:split]
        X_cal, y_cal = X.iloc[split:], y.iloc[split:]

        self.base.fit(X_base, y_base)
        cal_output = self.base.predict(X_cal)
        y_arr = y_cal.to_numpy(dtype=float)
        lower = np.asarray(cal_output.lower, dtype=float)
        upper = np.asarray(cal_output.upper, dtype=float)

        raw_hits = (y_arr >= lower) & (y_arr <= upper)
        self._calibration_raw_coverage = float(np.mean(raw_hits))

        # Non-conformity score: 0 inside the interval, positive miss
        # magnitude when outside. The target quantile of these scores
        # is the amount we must widen by to achieve target coverage.
        scores = np.maximum.reduce([lower - y_arr, y_arr - upper, np.zeros_like(y_arr)])

        # Finite-sample correction (Vovk-style): use (⌈(n+1)·α⌉ / n)-th
        # quantile rather than the plain α-quantile. With α=coverage this
        # bumps the quantile level slightly to guarantee ≥ target coverage.
        adjusted_level = min(
            1.0,
            np.ceil((len(scores) + 1) * self.target_coverage) / len(scores),
        )
        self._width = float(np.quantile(scores, adjusted_level))
        self._calibration_n = int(len(scores))

        # Refit on all available data so inference uses everything.
        self.base.fit(X, y)
        logger.info(
            "Conformal calibration: raw coverage=%.2f, widening=%.3f, target=%.2f, n=%d",
            self._calibration_raw_coverage,
            self._width,
            self.target_coverage,
            self._calibration_n,
        )
        return self

    def predict(self, X: pd.DataFrame) -> ForecastOutput:
        base_output = self.base.predict(X)
        predicted = np.asarray(base_output.predicted, dtype=float)
        lower = np.asarray(base_output.lower, dtype=float) - self._width
        upper = np.asarray(base_output.upper, dtype=float) + self._width

        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, predicted)
        predicted = np.maximum(predicted, 0.0)
        return ForecastOutput(
            predicted=predicted,
            lower=lower,
            upper=upper,
            quantiles=base_output.quantiles,
        )

    @property
    def calibration_summary(self) -> dict[str, float | int]:
        return {
            "target_coverage": self.target_coverage,
            "calibration_frac": self.calibration_frac,
            "calibration_n": self._calibration_n,
            "calibration_raw_coverage": self._calibration_raw_coverage,
            "width_adjustment": self._width,
        }
