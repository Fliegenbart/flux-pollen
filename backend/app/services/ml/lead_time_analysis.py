"""Lead-time analysis from persisted backtest points.

The pitch question is: for a historically observed pollen peak, how
many days *in advance* did our model first signal "this is coming",
compared to the two free baselines (Persistence and Seasonal-Naive)?

A peak is defined as a target_date whose observed concentration is in
the top ``peak_percentile`` of the fold sample for that scope. For
each peak we scan backwards through the folds that forecast the same
target_date and find the earliest forecast_date where each predictor
(model / persistence / seasonal) first reported a value ≥ the event
threshold. The difference, in days, is the lead time.

This is the canonical ops-side measurement of forecast value in
epi/hub settings — if the model signals earlier than the naïve
alternatives, that's the operational window in which a media buy,
inventory reorder, or search-bid change actually has lever arm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.database import BacktestPoint, BacktestRun

__all__ = [
    "PeakLeadTime",
    "compute_lead_times_for_scope",
    "compute_lead_time_summary",
]


@dataclass(frozen=True)
class PeakLeadTime:
    run_id: str
    target_date: datetime
    observed_value: float
    event_threshold: float
    model_first_signal: datetime | None
    model_lead_days: int | None
    persistence_first_signal: datetime | None
    persistence_lead_days: int | None
    seasonal_first_signal: datetime | None
    seasonal_lead_days: int | None
    folds_considered: int


def compute_lead_times_for_scope(
    db: Session,
    *,
    pollen_type: str,
    region_code: str,
    horizon_days: int | None = None,
    model_version_prefix: str | None = None,
    peak_percentile: float = 0.90,
    event_threshold_ratio: float = 0.5,
    top_n_peaks: int = 10,
) -> list[PeakLeadTime]:
    """Return the lead-time record for the most striking historical peaks.

    If ``horizon_days`` is None, we pool points from *all* recent runs
    for the (pollen, region) — one run per horizon — so a single peak
    can have several fold predictions at different forecast_dates,
    which is what lets lead time actually span more than one step.

    Parameters
    ----------
    peak_percentile
        What counts as a "peak": target_dates whose observed y is at or
        above this quantile of the run's observed-y distribution.
    event_threshold_ratio
        A forecast "signals" the peak when its point prediction crosses
        ``event_threshold_ratio × observed_value``. 0.5 means "any run
        that says at least half of the eventual peak" — generous enough
        to avoid nitpicking near the top, strict enough to be a real
        signal. Baselines use the same threshold.
    top_n_peaks
        Keep only the N biggest peaks, sorted descending by observed_y.
    """
    run_query = (
        db.query(BacktestRun)
        .filter(
            BacktestRun.pollen_type == pollen_type,
            BacktestRun.region_code == region_code,
        )
        .order_by(BacktestRun.created_at.desc())
    )
    if horizon_days is not None:
        run_query = run_query.filter(BacktestRun.horizon_days == horizon_days)
    if model_version_prefix:
        run_query = run_query.filter(BacktestRun.model_version.like(f"{model_version_prefix}%"))
    runs = run_query.all()
    if not runs:
        return []

    # For each horizon, keep only the most recent run of the scope so
    # we don't double-count folds across re-runs of the same setup.
    selected: dict[int, BacktestRun] = {}
    for run in runs:
        selected.setdefault(int(run.horizon_days), run)
    run_ids = [r.run_id for r in selected.values()]

    points = (
        db.query(BacktestPoint)
        .filter(BacktestPoint.run_id.in_(run_ids))
        .order_by(BacktestPoint.date.asc())
        .all()
    )
    if not points:
        return []

    frame = pd.DataFrame(
        {
            "target_date": [pd.Timestamp(p.date) for p in points],
            "real_value": [float(p.real_value) if p.real_value is not None else np.nan for p in points],
            "predicted_value": [float(p.predicted_value) if p.predicted_value is not None else np.nan for p in points],
            "persistence": [float(p.baseline_persistence) if p.baseline_persistence is not None else np.nan for p in points],
            "seasonal": [float(p.baseline_seasonal) if p.baseline_seasonal is not None else np.nan for p in points],
            "extra": [p.extra or {} for p in points],
        }
    )
    frame["forecast_date"] = [
        pd.Timestamp(entry.get("forecast_date")) if isinstance(entry, dict) and entry.get("forecast_date")
        else pd.NaT
        for entry in frame["extra"]
    ]
    frame = frame.dropna(subset=["real_value", "forecast_date"])
    if frame.empty:
        return []

    # Peak detection on the unique (target_date, real_value) pairs —
    # otherwise the same peak would appear once per horizon run and
    # skew the quantile threshold.
    unique_targets = (
        frame.drop_duplicates(subset=["target_date"])[["target_date", "real_value"]]
        .reset_index(drop=True)
    )
    threshold_quantile = float(np.quantile(unique_targets["real_value"], peak_percentile))
    peaks = (
        unique_targets.loc[unique_targets["real_value"] >= threshold_quantile]
        .sort_values("real_value", ascending=False)
        .head(int(top_n_peaks))
        .copy()
    )

    peak_summaries: list[PeakLeadTime] = []
    for _, peak_row in peaks.iterrows():
        target_date = peak_row["target_date"]
        observed = float(peak_row["real_value"])
        event_thresh = observed * float(event_threshold_ratio)

        # Collect every fold whose forecast_date precedes this peak and
        # whose target_date == peak.target_date. Because the backtester
        # uses a direct-h step, different folds targeting the same date
        # are separated by the walk-forward stride.
        window = frame.loc[frame["target_date"] == target_date].copy()
        window = window.sort_values("forecast_date")
        if window.empty:
            continue

        def _first_crossing(col: str) -> datetime | None:
            match = window.loc[window[col] >= event_thresh, "forecast_date"]
            if match.empty:
                return None
            return match.iloc[0].to_pydatetime()

        model_signal = _first_crossing("predicted_value")
        persistence_signal = _first_crossing("persistence")
        seasonal_signal = _first_crossing("seasonal")
        target_py = target_date.to_pydatetime()

        def _lead(signal: datetime | None) -> int | None:
            if signal is None:
                return None
            return max(0, (target_py - signal).days)

        peak_summaries.append(
            PeakLeadTime(
                run_id=run.run_id,
                target_date=target_py,
                observed_value=round(observed, 2),
                event_threshold=round(event_thresh, 2),
                model_first_signal=model_signal,
                model_lead_days=_lead(model_signal),
                persistence_first_signal=persistence_signal,
                persistence_lead_days=_lead(persistence_signal),
                seasonal_first_signal=seasonal_signal,
                seasonal_lead_days=_lead(seasonal_signal),
                folds_considered=int(len(window)),
            )
        )
    return peak_summaries


def compute_lead_time_summary(peaks: list[PeakLeadTime]) -> dict[str, Any]:
    """Aggregate numbers suitable for a pitch headline."""
    if not peaks:
        return {"n_peaks": 0}
    model = [p.model_lead_days for p in peaks if p.model_lead_days is not None]
    pers = [p.persistence_lead_days for p in peaks if p.persistence_lead_days is not None]
    seas = [p.seasonal_lead_days for p in peaks if p.seasonal_lead_days is not None]
    return {
        "n_peaks": len(peaks),
        "model_lead_days_median": float(np.median(model)) if model else None,
        "model_lead_days_mean": round(float(np.mean(model)), 1) if model else None,
        "persistence_lead_days_median": float(np.median(pers)) if pers else None,
        "seasonal_lead_days_median": float(np.median(seas)) if seas else None,
        "model_advantage_vs_persistence_days": (
            round(float(np.mean(model) - np.mean(pers)), 1)
            if model and pers
            else None
        ),
    }
