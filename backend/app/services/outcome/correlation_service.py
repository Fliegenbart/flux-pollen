"""Correlation service: customer outcome ↔ pollen signal.

This is the first piece of product logic that translates a pollen
forecast into something a Marketing manager cares about. For a given
(brand, product, metric, region) it pulls the uploaded weekly outcome
series, reconstructs the daily pollen-observation series for the same
period, aggregates pollen to the same weekly grid, and reports:

- Pearson correlation between outcome and pollen at candidate lags.
- The **best lag** — i.e. how many days the pollen signal *leads*
  the outcome on average.
- A lift estimate at the 80th percentile of pollen concentration:
  how much higher did the outcome metric run during high-pollen
  weeks vs. low-pollen weeks? This is a simple but honest proxy for
  the commercial elasticity a customer is paying for.

The output is deliberately compact and JSON-shaped so the frontend
can render it as a single infographic without a second round-trip.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.database import OutcomeObservation, PollenObservation

logger = logging.getLogger(__name__)

__all__ = ["OutcomeCorrelationService", "CorrelationResult"]


@dataclass(frozen=True)
class CorrelationResult:
    brand: str
    product: str
    region_code: str
    pollen_type: str
    metric: str
    n_weeks: int
    lag_curve: list[dict[str, float]]  # list of {"lag_days": int, "pearson": float}
    best_lag_days: int
    best_pearson: float
    lift_high_vs_low_pct: float
    high_weeks: int
    low_weeks: int
    outcome_series: list[dict[str, Any]]
    pollen_series: list[dict[str, Any]]


class OutcomeCorrelationService:
    def __init__(self, db: Session):
        self.db = db

    def compute(
        self,
        *,
        brand: str,
        product: str,
        region_code: str,
        pollen_type: str,
        metric: str,
        lag_days_range: range = range(-21, 22, 1),
    ) -> CorrelationResult:
        outcome_frame = self._load_outcome(
            brand=brand,
            product=product,
            region_code=region_code,
            metric=metric,
        )
        if outcome_frame.empty:
            raise ValueError(
                f"No outcome data for brand={brand!r}, product={product!r}, "
                f"region={region_code!r}, metric={metric!r}."
            )

        pollen_daily = self._load_pollen(
            region_code=region_code,
            pollen_type=pollen_type,
            start=outcome_frame["week_start"].min() - pd.Timedelta(days=30),
            end=outcome_frame["week_start"].max() + pd.Timedelta(days=7),
        )
        if pollen_daily.empty:
            raise ValueError(
                f"No pollen data for region={region_code!r}, pollen_type={pollen_type!r}."
            )

        pollen_weekly = self._aggregate_pollen_weekly(pollen_daily)

        joined = outcome_frame.merge(
            pollen_weekly.rename(columns={"week_start": "week_start"}),
            on="week_start",
            how="inner",
        )
        if len(joined) < 6:
            raise ValueError(
                f"Only {len(joined)} overlapping weeks — need at least 6 for a credible correlation."
            )

        lag_curve = self._lag_curve(outcome_frame, pollen_daily, lag_days_range)
        best = max(lag_curve, key=lambda item: item["pearson"])

        lift, high_n, low_n = self._high_vs_low_lift(joined)

        outcome_series = [
            {"week_start": ws.isoformat(), "value": float(v)}
            for ws, v in zip(outcome_frame["week_start"], outcome_frame["value"])
        ]
        pollen_series = [
            {"week_start": ws.isoformat(), "concentration": float(v)}
            for ws, v in zip(pollen_weekly["week_start"], pollen_weekly["pollen_mean"])
        ]

        return CorrelationResult(
            brand=brand,
            product=product,
            region_code=region_code,
            pollen_type=pollen_type,
            metric=metric,
            n_weeks=int(len(joined)),
            lag_curve=lag_curve,
            best_lag_days=int(best["lag_days"]),
            best_pearson=float(best["pearson"]),
            lift_high_vs_low_pct=float(lift),
            high_weeks=int(high_n),
            low_weeks=int(low_n),
            outcome_series=outcome_series,
            pollen_series=pollen_series,
        )

    # ----------------------------------------------------------------- #
    # Data loaders
    # ----------------------------------------------------------------- #

    def _load_outcome(
        self,
        *,
        brand: str,
        product: str,
        region_code: str,
        metric: str,
    ) -> pd.DataFrame:
        rows = (
            self.db.query(
                OutcomeObservation.window_start,
                OutcomeObservation.metric_value,
            )
            .filter(
                OutcomeObservation.brand == brand.lower(),
                OutcomeObservation.product == product.lower(),
                OutcomeObservation.region_code == region_code.upper(),
                OutcomeObservation.metric_name == metric.lower(),
            )
            .order_by(OutcomeObservation.window_start.asc())
            .all()
        )
        if not rows:
            return pd.DataFrame(columns=["week_start", "value"])
        frame = pd.DataFrame.from_records(rows, columns=["week_start", "value"])
        frame["week_start"] = pd.to_datetime(frame["week_start"]).dt.normalize()
        frame["value"] = frame["value"].astype(float)
        # If there are duplicates by week, sum them — common with multi-channel data.
        frame = frame.groupby("week_start", as_index=False)["value"].sum()
        frame = frame.sort_values("week_start").reset_index(drop=True)
        return frame

    def _load_pollen(
        self,
        *,
        region_code: str,
        pollen_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        rows = (
            self.db.query(
                func.date(PollenObservation.from_time).label("datum"),
                func.avg(PollenObservation.concentration).label("concentration"),
            )
            .filter(
                PollenObservation.region_code == region_code.upper(),
                PollenObservation.pollen_type == pollen_type.lower(),
                PollenObservation.from_time >= start.to_pydatetime(),
                PollenObservation.from_time <= end.to_pydatetime(),
            )
            .group_by(func.date(PollenObservation.from_time))
            .order_by(func.date(PollenObservation.from_time))
            .all()
        )
        if not rows:
            return pd.DataFrame(columns=["datum", "concentration"])
        frame = pd.DataFrame.from_records(rows, columns=["datum", "concentration"])
        frame["datum"] = pd.to_datetime(frame["datum"]).dt.normalize()
        frame["concentration"] = frame["concentration"].astype(float)
        return frame

    @staticmethod
    def _aggregate_pollen_weekly(daily: pd.DataFrame) -> pd.DataFrame:
        frame = daily.copy()
        frame["week_start"] = frame["datum"] - pd.to_timedelta(frame["datum"].dt.weekday, unit="D")
        weekly = (
            frame.groupby("week_start", as_index=False)
            .agg(pollen_mean=("concentration", "mean"))
            .sort_values("week_start")
            .reset_index(drop=True)
        )
        return weekly

    # ----------------------------------------------------------------- #
    # Analysis
    # ----------------------------------------------------------------- #

    @staticmethod
    def _lag_curve(
        outcome_frame: pd.DataFrame,
        pollen_daily: pd.DataFrame,
        lag_days_range: range,
    ) -> list[dict[str, float]]:
        """Correlation for a grid of lag values in days.

        Positive lag = pollen leads outcome by that many days.
        For each lag we shift the daily pollen series forward by the
        lag, re-aggregate to weeks, and join to the outcome. Using
        daily pollen rather than weekly lets us evaluate sub-weekly
        leads (typical for fast-moving search behaviour).
        """
        results: list[dict[str, float]] = []
        outcome_frame = outcome_frame.set_index("week_start").sort_index()

        for lag in lag_days_range:
            shifted = pollen_daily.copy()
            shifted["datum"] = shifted["datum"] + pd.Timedelta(days=int(lag))
            shifted["week_start"] = shifted["datum"] - pd.to_timedelta(
                shifted["datum"].dt.weekday, unit="D"
            )
            weekly = shifted.groupby("week_start")["concentration"].mean()
            joined = outcome_frame.join(weekly.rename("pollen_mean"), how="inner")
            if len(joined) < 6:
                results.append({"lag_days": int(lag), "pearson": float("nan")})
                continue
            v = joined["value"].to_numpy(dtype=float)
            p = joined["pollen_mean"].to_numpy(dtype=float)
            if np.std(v) == 0 or np.std(p) == 0:
                results.append({"lag_days": int(lag), "pearson": 0.0})
                continue
            pearson = float(np.corrcoef(v, p)[0, 1])
            results.append({"lag_days": int(lag), "pearson": round(pearson, 4)})
        return results

    @staticmethod
    def _high_vs_low_lift(joined: pd.DataFrame) -> tuple[float, int, int]:
        """Lift ≈ mean(outcome | pollen in top quartile) / mean(outcome | bottom quartile) − 1."""
        if len(joined) < 6:
            return 0.0, 0, 0
        q_high = float(np.quantile(joined["pollen_mean"], 0.75))
        q_low = float(np.quantile(joined["pollen_mean"], 0.25))
        high = joined.loc[joined["pollen_mean"] >= q_high, "value"].astype(float)
        low = joined.loc[joined["pollen_mean"] <= q_low, "value"].astype(float)
        if len(high) == 0 or len(low) == 0 or float(low.mean()) == 0:
            return 0.0, int(len(high)), int(len(low))
        lift = float(high.mean()) / float(low.mean()) - 1.0
        return round(lift * 100.0, 2), int(len(high)), int(len(low))
