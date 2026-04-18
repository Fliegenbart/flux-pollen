"""PollenCast SQLAlchemy model definitions.

Ten business tables plus AuditLog. Schema evolution is managed via Alembic; this
module is the single source of truth for table definitions.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class PollenData(Base):
    """DWD-Pollenflug-Index (0–3) pro Bundesland × Pollenart × Tag."""
    __tablename__ = "pollen_data"

    id = Column(Integer, primary_key=True, index=True)
    datum = Column(DateTime, nullable=False, index=True)
    available_time = Column(DateTime, nullable=True, index=True)
    region_code = Column(String(2), nullable=False, index=True)
    pollen_type = Column(String(32), nullable=False, index=True)
    pollen_index = Column(Float, nullable=False)
    source = Column(String(32), nullable=False, default="DWD")
    created_at = Column(DateTime, default=_utc_now, nullable=False, index=True)

    __table_args__ = (
        Index("idx_pollen_region_date", "region_code", "datum"),
        Index("idx_pollen_type_date", "pollen_type", "datum"),
        UniqueConstraint(
            "region_code",
            "pollen_type",
            "datum",
            name="uq_pollen_region_type_date",
        ),
    )


class PollenObservation(Base):
    """Station-level Pollenkonzentrationsmessungen (count/m³, zeitlich granular).

    Zielquellen: ePIN Bayern (`PomoAI`/`V09_21`-Algorithmen, 3-Stunden-Buckets,
    Automaten + Hirst-Fallen) und perspektivisch weitere Monitoring-Netzwerke.
    Parallel zu ``pollen_data`` (DWD-Index 0–3, regional/tagesweise) — hier ist
    die Feingranularität das Asset.
    """
    __tablename__ = "pollen_observations"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(String(16), nullable=False, index=True)
    station_name = Column(String(64), nullable=True)
    region_code = Column(String(2), nullable=False, index=True)
    pollen_type = Column(String(32), nullable=False, index=True)
    # Messfenster im natürlichen Quelltakt (typisch 3 Stunden bei ePIN).
    from_time = Column(DateTime, nullable=False, index=True)
    to_time = Column(DateTime, nullable=False)
    concentration = Column(Float, nullable=False)
    algorithm = Column(String(32), nullable=True)
    source_network = Column(String(16), nullable=False, default="ePIN")
    available_time = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=_utc_now, nullable=False, index=True)

    __table_args__ = (
        Index("idx_pollen_obs_station_type_from", "station_id", "pollen_type", "from_time"),
        Index("idx_pollen_obs_region_type_from", "region_code", "pollen_type", "from_time"),
        UniqueConstraint(
            "station_id",
            "pollen_type",
            "from_time",
            name="uq_pollen_obs_station_type_window",
        ),
    )


class PollenForecast(Base):
    """Probabilistischer Pollen-Forecast (Punkt + Quantile) pro Bundesland × Pollenart × Horizon."""
    __tablename__ = "pollen_forecast"

    id = Column(Integer, primary_key=True, index=True)
    pollen_type = Column(String(32), nullable=False, index=True)
    region_code = Column(String(2), nullable=False, index=True)
    horizon_days = Column(Integer, nullable=False, index=True)
    forecast_date = Column(DateTime, nullable=False, index=True)
    target_date = Column(DateTime, nullable=False, index=True)
    predicted_index = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    confidence_label = Column(String(16), nullable=True)
    model_version = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False, index=True)

    __table_args__ = (
        Index("idx_forecast_scope_target", "pollen_type", "region_code", "horizon_days", "target_date"),
        Index("idx_forecast_scope_created", "pollen_type", "region_code", "horizon_days", "created_at"),
    )


class WeatherData(Base):
    """Tägliche Wetterfeatures pro Stadt/Bundesland (DWD oder OpenWeather)."""
    __tablename__ = "weather_data"

    id = Column(Integer, primary_key=True, index=True)
    datum = Column(DateTime, nullable=False, index=True)
    available_time = Column(DateTime, nullable=True, index=True)
    city = Column(String(64), nullable=False)
    region_code = Column(String(2), nullable=True, index=True)
    temperatur = Column(Float, nullable=True)
    gefuehlte_temperatur = Column(Float, nullable=True)
    luftfeuchtigkeit = Column(Float, nullable=True)
    luftdruck = Column(Float, nullable=True)
    wind_geschwindigkeit = Column(Float, nullable=True)
    wolken = Column(Float, nullable=True)
    niederschlag_wahrscheinlichkeit = Column(Float, nullable=True)
    regen_mm = Column(Float, nullable=True)
    taupunkt = Column(Float, nullable=True)
    data_type = Column(String(32), nullable=False, default="CURRENT")
    forecast_run_timestamp = Column(DateTime, nullable=True, index=True)
    forecast_run_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False)

    __table_args__ = (
        Index("idx_weather_date_city", "datum", "city"),
        Index("idx_weather_region_date", "region_code", "datum"),
        Index("idx_weather_data_type", "data_type"),
    )


class SchoolHolidays(Base):
    """Schulferien-Kalender pro Bundesland."""
    __tablename__ = "school_holidays"

    id = Column(Integer, primary_key=True, index=True)
    bundesland = Column(String(2), nullable=False, index=True)
    ferien_typ = Column(String(32), nullable=False)
    start_datum = Column(DateTime, nullable=False)
    end_datum = Column(DateTime, nullable=False)
    jahr = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False)

    __table_args__ = (
        Index("idx_holidays_dates", "start_datum", "end_datum"),
        Index("idx_holidays_bundesland_jahr", "bundesland", "jahr"),
    )


class BacktestRun(Base):
    """Persistierter Walk-Forward-Backtest-Lauf pro (Pollenart × Bundesland × Horizon)."""
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), unique=True, nullable=False, index=True)
    status = Column(String(16), nullable=False, default="success", index=True)
    pollen_type = Column(String(32), nullable=False, index=True)
    region_code = Column(String(2), nullable=True, index=True)
    horizon_days = Column(Integer, nullable=False, default=7)
    min_train_points = Column(Integer, nullable=True)
    parameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    baseline_metrics = Column(JSON, nullable=True)
    improvement_vs_baselines = Column(JSON, nullable=True)
    model_version = Column(String(64), nullable=True)
    chart_points = Column(Integer, nullable=True, default=0)
    created_at = Column(DateTime, default=_utc_now, nullable=False, index=True)

    __table_args__ = (
        Index("idx_backtest_scope_created", "pollen_type", "region_code", "horizon_days", "created_at"),
    )


class BacktestPoint(Base):
    """Einzelner Zeitpunkt eines Backtest-Laufs (Charts/Audit)."""
    __tablename__ = "backtest_points"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("backtest_runs.run_id"), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    region_code = Column(String(2), nullable=True, index=True)
    real_value = Column(Float, nullable=True)
    predicted_value = Column(Float, nullable=True)
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    baseline_persistence = Column(Float, nullable=True)
    baseline_seasonal = Column(Float, nullable=True)
    extra = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False)

    backtest_run = relationship("BacktestRun")

    __table_args__ = (
        Index("idx_backtest_points_run_date", "run_id", "date"),
    )


class ForecastAccuracyLog(Base):
    """Tägliches Monitoring: Forecast vs. Ist (Pollenart × Horizon × Fenster)."""
    __tablename__ = "forecast_accuracy_log"

    id = Column(Integer, primary_key=True, index=True)
    computed_at = Column(DateTime, default=_utc_now, nullable=False, index=True)
    pollen_type = Column(String(32), nullable=False, index=True)
    region_code = Column(String(2), nullable=True, index=True)
    horizon_days = Column(Integer, nullable=False, default=7)
    window_days = Column(Integer, nullable=False, default=14)
    samples = Column(Integer, nullable=False)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    wis = Column(Float, nullable=True)
    correlation = Column(Float, nullable=True)
    drift_detected = Column(Boolean, default=False, nullable=False)
    details = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_accuracy_scope_computed", "pollen_type", "region_code", "horizon_days", "computed_at"),
    )


class SourceNowcastSnapshot(Base):
    """Append-only Snapshots von Rohbeobachtungen für Point-in-Time-Feature-Bau."""
    __tablename__ = "source_nowcast_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(String(32), nullable=False, index=True)
    signal_id = Column(String(64), nullable=False, index=True)
    region_code = Column(String(2), nullable=True, index=True)
    reference_date = Column(DateTime, nullable=False, index=True)
    effective_available_time = Column(DateTime, nullable=False, index=True)
    raw_value = Column(Float, nullable=False)
    snapshot_captured_at = Column(DateTime, default=_utc_now, nullable=False, index=True)
    timing_provenance = Column(String(32), nullable=False)
    metadata_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False)

    __table_args__ = (
        Index("idx_nowcast_source_ref", "source_id", "reference_date"),
        Index("idx_nowcast_signal_region", "signal_id", "region_code"),
    )


class User(Base):
    """Minimale User-Tabelle für Backoffice/Analyst-Logins."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(16), nullable=False, default="user")
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=_utc_now, nullable=False)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)


class AuditLog(Base):
    """Audit-Trail für Logins, Logouts und Admin-Aktionen."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=_utc_now, nullable=False, index=True)
    user = Column(String(255), nullable=True, index=True)
    action = Column(String(64), nullable=False)
    entity_type = Column(String(64), nullable=True)
    entity_id = Column(String(64), nullable=True)
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    reason = Column(String(255), nullable=True)
    ip_address = Column(String(64), nullable=True)


class OutcomeObservation(Base):
    """Customer-provided commercial outcome observations.

    One row per (brand, product, region_code, metric_name, window_start,
    window_end, source_label). "Window" is typically a calendar week.
    Added in Phase 5a for the Hexal/Lorano pilot; designed to generalize
    to any OTC brand × metric without schema changes.
    """
    __tablename__ = "outcome_observations"

    id = Column(Integer, primary_key=True, index=True)
    brand = Column(String(64), nullable=False, index=True, default="hexal")
    product = Column(String(64), nullable=False, index=True)
    region_code = Column(String(2), nullable=False, index=True)
    window_start = Column(DateTime, nullable=False, index=True)
    window_end = Column(DateTime, nullable=False, index=True)
    metric_name = Column(String(64), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(32), nullable=True)
    source_label = Column(String(64), nullable=False, default="customer_upload", index=True)
    channel = Column(String(64), nullable=True, index=True)
    campaign_id = Column(String(64), nullable=True, index=True)
    holdout_group = Column(String(32), nullable=True)
    confidence_hint = Column(Float, nullable=True)
    metadata_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False, index=True)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "window_start",
            "window_end",
            "brand",
            "product",
            "region_code",
            "metric_name",
            "source_label",
            name="uq_outcome_observation",
        ),
        Index("idx_outcome_brand_window", "brand", "window_start"),
        Index("idx_outcome_metric_window", "metric_name", "window_start"),
        Index("idx_outcome_region_product", "region_code", "product"),
    )


class UploadHistory(Base):
    """Protokoll für Kunden-CSV-Uploads (z. B. Sell-Through-Ground-Truth)."""
    __tablename__ = "upload_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    upload_type = Column(String(32), nullable=False)
    file_format = Column(String(8), nullable=True)
    row_count = Column(Integer, nullable=True)
    date_range_start = Column(DateTime, nullable=True)
    date_range_end = Column(DateTime, nullable=True)
    status = Column(String(16), default="success", nullable=False)
    error_message = Column(String(512), nullable=True)
    summary = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False, index=True)
