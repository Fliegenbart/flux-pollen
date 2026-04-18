"""Feature engineering for pollen forecasts.

Reads the two data sources produced by Phase 3 — station-level ePIN
observations and capital-city weather — and assembles a daily feature
panel with strict Point-in-Time semantics: every feature value at date
``t`` is resolvable from data whose ``available_time <= t``.

Keeping this module small on purpose. The forecast is driven by four
families:

1. **Target history**: lags and short rolling windows on the daily mean
   concentration of the target pollen.
2. **Weather**: temperature, precipitation, humidity, wind — averaged
   across Bayern capitals and carried at lags 0/1/3.
3. **Calendar**: sin/cos day-of-year, weekday, holiday fraction.
4. **Cross-pollen**: lagged means of the other species that biologically
   lead the target (Hasel → Erle → Birke → Gräser chain).

We do not add neighbor-state features, interaction terms, or model
ensembles of feature subsets. Those belong in Phase 5 once we have
enough real seasons to tell whether they carry signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.database import PollenObservation, SchoolHolidays, WeatherData

__all__ = [
    "FeatureBuildConfig",
    "build_daily_panel",
    "build_target_features",
    "build_weather_features",
    "build_calendar_features",
    "build_cross_pollen_features",
    "assemble_training_frame",
    "CROSS_POLLEN_PARTNERS",
]


TARGET_LAGS: tuple[int, ...] = (1, 2, 3, 5, 7)
TARGET_ROLLING_WINDOWS: tuple[int, ...] = (3, 7)
WEATHER_LAGS: tuple[int, ...] = (0, 1, 3)
CROSS_POLLEN_LAGS: tuple[int, ...] = (7, 14)

# Biologically-informed lead/lag chain. For each target, which other
# species tend to move first? Hasel leads Erle, Erle leads Birke, etc.
# Keep the list short — more partners just inflate feature count.
CROSS_POLLEN_PARTNERS: dict[str, tuple[str, ...]] = {
    "hasel": (),
    "erle": ("hasel",),
    "esche": ("erle", "hasel"),
    "birke": ("erle", "esche"),
    "graeser": ("birke", "esche"),
    "roggen": ("graeser",),
    "beifuss": ("graeser",),
    "ambrosia": ("beifuss", "graeser"),
}


@dataclass(frozen=True)
class FeatureBuildConfig:
    region_code: str
    pollen_type: str
    start_date: datetime
    end_date: datetime
    target_lags: tuple[int, ...] = TARGET_LAGS
    target_rolling_windows: tuple[int, ...] = TARGET_ROLLING_WINDOWS
    weather_lags: tuple[int, ...] = WEATHER_LAGS
    cross_pollen_lags: tuple[int, ...] = CROSS_POLLEN_LAGS


# --------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------- #


def _load_daily_pollen_means(
    db: Session,
    *,
    region_code: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Daily mean concentration per pollen_type across all stations in a region.

    Aggregation: the 3-hour buckets from a single day are already a station's
    daily mean curve; we take the mean of those buckets per day per station,
    then the mean across stations → a single daily value per (region, pollen).
    """
    rows = (
        db.query(
            PollenObservation.from_time,
            PollenObservation.station_id,
            PollenObservation.pollen_type,
            PollenObservation.concentration,
        )
        .filter(
            PollenObservation.region_code == region_code,
            PollenObservation.from_time >= start,
            PollenObservation.from_time <= end,
        )
        .all()
    )
    if not rows:
        return pd.DataFrame(columns=["datum", "pollen_type", "concentration"])

    frame = pd.DataFrame.from_records(
        rows, columns=["from_time", "station_id", "pollen_type", "concentration"]
    )
    frame["datum"] = pd.to_datetime(frame["from_time"]).dt.normalize()
    station_daily = frame.groupby(["datum", "station_id", "pollen_type"], as_index=False)[
        "concentration"
    ].mean()
    daily = station_daily.groupby(["datum", "pollen_type"], as_index=False)[
        "concentration"
    ].mean()
    return daily.sort_values(["pollen_type", "datum"]).reset_index(drop=True)


def _load_daily_weather(
    db: Session,
    *,
    region_code: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Daily weather features for a region.

    We keep ``DAILY_OBSERVATION`` rows only — the CURRENT/FORECAST rows
    would leak future information. For future horizons the feature
    builder uses the last observed values held constant; Phase 5 will
    plug in MOSMIX forecasts with a strict ``available_time`` guard.
    """
    rows = (
        db.query(
            WeatherData.datum,
            WeatherData.temperatur,
            WeatherData.luftfeuchtigkeit,
            WeatherData.wind_geschwindigkeit,
            WeatherData.regen_mm,
        )
        .filter(
            WeatherData.region_code == region_code,
            WeatherData.datum >= start,
            WeatherData.datum <= end,
            WeatherData.data_type == "DAILY_OBSERVATION",
        )
        .all()
    )
    if not rows:
        return pd.DataFrame(columns=["datum", "temperatur", "luftfeuchtigkeit", "wind_geschwindigkeit", "regen_mm"])

    frame = pd.DataFrame.from_records(
        rows, columns=["datum", "temperatur", "luftfeuchtigkeit", "wind_geschwindigkeit", "regen_mm"]
    )
    frame["datum"] = pd.to_datetime(frame["datum"]).dt.normalize()
    frame = (
        frame.groupby("datum", as_index=False)
        .agg(
            temperatur=("temperatur", "mean"),
            luftfeuchtigkeit=("luftfeuchtigkeit", "mean"),
            wind_geschwindigkeit=("wind_geschwindigkeit", "mean"),
            regen_mm=("regen_mm", "sum"),
        )
        .sort_values("datum")
        .reset_index(drop=True)
    )
    return frame


def _load_holidays_fraction(
    db: Session,
    *,
    region_code: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fraction of school-holiday days — here 0 or 1 per day since we're
    single-region. Left as a fraction so Phase 5 can average across
    neighbouring states without a schema change.
    """
    days = pd.date_range(start=start, end=end, freq="D", normalize=True)
    flag = np.zeros(len(days), dtype=float)
    rows = (
        db.query(SchoolHolidays.start_datum, SchoolHolidays.end_datum)
        .filter(SchoolHolidays.bundesland == region_code)
        .all()
    )
    for start_dt, end_dt in rows:
        if start_dt is None or end_dt is None:
            continue
        start_dt = pd.Timestamp(start_dt).normalize()
        end_dt = pd.Timestamp(end_dt).normalize()
        mask = (days >= start_dt) & (days <= end_dt)
        flag[mask] = 1.0
    return pd.DataFrame({"datum": days, "holiday_flag": flag})


# --------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------- #


def build_target_features(
    target_series: pd.Series,
    *,
    lags: Sequence[int] = TARGET_LAGS,
    rolling_windows: Sequence[int] = TARGET_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Lagged values + rolling mean/slope of the target concentration."""
    frame = pd.DataFrame(index=target_series.index)
    frame["y"] = target_series.astype(float)
    for lag in lags:
        frame[f"y_lag{lag}"] = target_series.shift(int(lag))
    for window in rolling_windows:
        rolled = target_series.shift(1).rolling(window=int(window), min_periods=max(2, int(window) // 2))
        frame[f"y_roll{window}_mean"] = rolled.mean()
        # Slope: last-minus-first over the window, a cheap trend signal.
        frame[f"y_roll{window}_slope"] = (
            target_series.shift(1) - target_series.shift(int(window))
        ) / float(window)
    return frame


def build_weather_features(
    weather: pd.DataFrame,
    *,
    lags: Sequence[int] = WEATHER_LAGS,
) -> pd.DataFrame:
    """Weather features aligned to the pollen date index, with lags."""
    frame = weather.set_index("datum").sort_index()
    out = pd.DataFrame(index=frame.index)
    for col in ("temperatur", "luftfeuchtigkeit", "wind_geschwindigkeit", "regen_mm"):
        if col not in frame.columns:
            continue
        for lag in lags:
            suffix = "now" if int(lag) == 0 else f"lag{int(lag)}"
            out[f"wx_{col}_{suffix}"] = frame[col].shift(int(lag))
    return out


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclic-encoded day-of-year + weekend + two quarterly dummies.

    Day-of-year uses sin/cos to avoid the Dec→Jan discontinuity a linear
    integer would introduce. Weekend matters because Google search and
    OTC purchase patterns differ sharply at weekends; not strictly a
    pollen-biology feature but a downstream-signal proxy.
    """
    days = index.dayofyear.astype(float)
    scale = 2.0 * math.pi / 365.25
    return pd.DataFrame(
        {
            "cal_doy_sin": np.sin(days * scale),
            "cal_doy_cos": np.cos(days * scale),
            "cal_weekend": (index.dayofweek >= 5).astype(float),
        },
        index=index,
    )


def build_cross_pollen_features(
    daily_frame: pd.DataFrame,
    *,
    target_pollen: str,
    lags: Sequence[int] = CROSS_POLLEN_LAGS,
) -> pd.DataFrame:
    """Lagged mean concentrations of the known upstream species.

    ``daily_frame`` is expected to be wide: one column per pollen_type.
    Any partner that is not present in the frame is silently skipped —
    early in a rollout we often only have enough data for 3 or 4 species.
    """
    partners = CROSS_POLLEN_PARTNERS.get(target_pollen, ())
    out = pd.DataFrame(index=daily_frame.index)
    for partner in partners:
        if partner not in daily_frame.columns:
            continue
        for lag in lags:
            out[f"xp_{partner}_lag{int(lag)}"] = daily_frame[partner].shift(int(lag))
    return out


# --------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------- #


def build_daily_panel(db: Session, config: FeatureBuildConfig) -> pd.DataFrame:
    """Build one row per calendar day with the target + all feature blocks.

    Returned frame has a DatetimeIndex and column ``y`` plus all features.
    Rows prior to the maximum lag are kept but will contain NaNs — leave
    them for the caller to drop so the backtester can walk through them.
    """
    pollen = _load_daily_pollen_means(
        db,
        region_code=config.region_code,
        start=config.start_date,
        end=config.end_date,
    )
    if pollen.empty:
        return pd.DataFrame()
    wide = pollen.pivot(index="datum", columns="pollen_type", values="concentration")
    wide = wide.sort_index()

    if config.pollen_type not in wide.columns:
        raise KeyError(
            f"No observations for pollen_type={config.pollen_type!r} in region "
            f"{config.region_code!r} between {config.start_date} and {config.end_date}."
        )

    target_frame = build_target_features(
        wide[config.pollen_type],
        lags=config.target_lags,
        rolling_windows=config.target_rolling_windows,
    )
    weather = _load_daily_weather(
        db,
        region_code=config.region_code,
        start=config.start_date,
        end=config.end_date,
    )
    weather_features = build_weather_features(weather, lags=config.weather_lags)
    calendar_features = build_calendar_features(target_frame.index)
    cross_pollen_features = build_cross_pollen_features(
        wide, target_pollen=config.pollen_type, lags=config.cross_pollen_lags
    )

    holidays = _load_holidays_fraction(
        db,
        region_code=config.region_code,
        start=config.start_date,
        end=config.end_date,
    )
    holidays_indexed = holidays.set_index("datum")

    joined = target_frame.join(weather_features, how="left")
    joined = joined.join(calendar_features, how="left")
    joined = joined.join(cross_pollen_features, how="left")
    joined = joined.join(holidays_indexed, how="left")
    return joined


def assemble_training_frame(
    panel: pd.DataFrame,
    *,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Materialize ``X`` and ``y`` for a direct h-step forecast.

    We predict ``y(t + h)`` from features known at time ``t`` — a direct
    strategy, not iterated. This keeps feature semantics honest and
    avoids error propagation across horizons.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    panel = panel.copy().sort_index()
    panel["target"] = panel["y"].shift(-int(horizon))
    features = panel.drop(columns=["y", "target"])
    # Keep only fully-populated rows for training/prediction.
    keep = features.dropna().index.intersection(panel["target"].dropna().index)
    X = features.loc[keep]
    y = panel["target"].loc[keep]
    return X, y, pd.DatetimeIndex(keep)
