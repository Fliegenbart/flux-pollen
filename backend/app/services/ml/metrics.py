"""Probabilistic forecast metrics.

We evaluate every forecast on three axes:

1. **Point accuracy** — MAE, RMSE.
2. **Quantile calibration** — Pinball loss at each quantile and the
   Weighted Interval Score (WIS), the proper scoring rule used by the
   COVID-19 Forecast Hub, the German RKI epiforecast consortium, and
   the Hubverse initiative. WIS decomposes into sharpness + over/under-
   prediction penalties and reduces to pinball loss when you use a
   single quantile.
3. **Dispersion / coverage** — empirical coverage of the 80 % interval
   so we can spot over- or under-confident forecasts.

The WIS definition follows Bracher et al. 2021, *Evaluating epidemic
forecasts in an interval format*: for an observed ``y`` and a forecast
consisting of the median ``m`` plus interval pairs ``(L_α, U_α)`` at
levels ``α ∈ {α_1, ..., α_K}``,

    WIS = (1 / (K + 0.5)) · (
        0.5 · |y − m|
        + Σ_k  (α_k / 2) · IS_{α_k}(y, L_k, U_k)
    )

where ``IS_α`` is the standard interval score. The implementation here
supports the common Hubverse convention where the ``alpha`` parameter
is the *central interval* (so 0.8 ↦ 10th/90th percentiles).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mae",
    "rmse",
    "pinball_loss",
    "interval_score",
    "weighted_interval_score",
    "coverage",
]


def _as_float_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def mae(y_true, y_pred) -> float:
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pinball_loss(y_true, y_pred_quantile, quantile: float) -> float:
    """Pinball / quantile loss for a single quantile level ``q``.

    Lower is better. ``pinball_loss(y, y_hat, q=0.5) * 2 == MAE(y, y_hat)``
    — that identity is a useful gut-check when you're reading the numbers.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred_quantile)
    delta = y_true - y_pred
    loss = np.where(delta >= 0.0, quantile * delta, (quantile - 1.0) * delta)
    return float(np.mean(loss))


def interval_score(
    y_true,
    lower,
    upper,
    alpha: float,
) -> float:
    """Standard interval score for a central (1 − α) prediction interval.

    ``IS_α(y, L, U) = (U − L) + (2/α) · (L − y)·𝟙[y<L] + (2/α) · (y − U)·𝟙[y>U]``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    y = _as_float_array(y_true)
    lo = _as_float_array(lower)
    hi = _as_float_array(upper)
    width = hi - lo
    under = np.maximum(lo - y, 0.0) * (2.0 / alpha)
    over = np.maximum(y - hi, 0.0) * (2.0 / alpha)
    return float(np.mean(width + under + over))


def weighted_interval_score(
    y_true,
    median,
    interval_bounds: dict[float, tuple[float | np.ndarray, float | np.ndarray]],
) -> float:
    """Weighted Interval Score (Bracher et al. 2021).

    ``interval_bounds`` maps α → (lower, upper) for each central interval
    you want to include. Typical choice: ``{0.2: (q10, q90), 0.5: (q25, q75)}``
    for a two-interval WIS.
    """
    if not interval_bounds:
        raise ValueError("interval_bounds must contain at least one (α, (L, U)) pair.")
    y = _as_float_array(y_true)
    med = _as_float_array(median)
    k = len(interval_bounds)
    total = 0.5 * np.abs(y - med)
    for alpha, (lo, hi) in interval_bounds.items():
        score = interval_score(y, lo, hi, alpha)
        total = total + (alpha / 2.0) * score
    return float(np.mean(total) / (k + 0.5))


def coverage(y_true, lower, upper) -> float:
    """Fraction of observations that fall inside the [lower, upper] interval."""
    y = _as_float_array(y_true)
    lo = _as_float_array(lower)
    hi = _as_float_array(upper)
    hits = (y >= lo) & (y <= hi)
    return float(np.mean(hits))
