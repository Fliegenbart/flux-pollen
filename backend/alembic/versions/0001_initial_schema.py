"""Initial PollenCast schema.

Creates all business tables that back the forecast pipeline:
pollen_data, pollen_forecast, weather_data, school_holidays,
backtest_runs, backtest_points, forecast_accuracy_log,
source_nowcast_snapshots, users, audit_logs, upload_history.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-04-18

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pollen_data",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("datum", sa.DateTime(), nullable=False),
        sa.Column("available_time", sa.DateTime(), nullable=True),
        sa.Column("region_code", sa.String(length=2), nullable=False),
        sa.Column("pollen_type", sa.String(length=32), nullable=False),
        sa.Column("pollen_index", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="DWD"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_pollen_data_id", "pollen_data", ["id"], unique=False)
    op.create_index("ix_pollen_data_datum", "pollen_data", ["datum"], unique=False)
    op.create_index("ix_pollen_data_available_time", "pollen_data", ["available_time"], unique=False)
    op.create_index("ix_pollen_data_region_code", "pollen_data", ["region_code"], unique=False)
    op.create_index("ix_pollen_data_pollen_type", "pollen_data", ["pollen_type"], unique=False)
    op.create_index("ix_pollen_data_created_at", "pollen_data", ["created_at"], unique=False)
    op.create_index("idx_pollen_region_date", "pollen_data", ["region_code", "datum"], unique=False)
    op.create_index("idx_pollen_type_date", "pollen_data", ["pollen_type", "datum"], unique=False)
    op.create_unique_constraint(
        "uq_pollen_region_type_date", "pollen_data", ["region_code", "pollen_type", "datum"]
    )

    op.create_table(
        "pollen_forecast",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("pollen_type", sa.String(length=32), nullable=False),
        sa.Column("region_code", sa.String(length=2), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("forecast_date", sa.DateTime(), nullable=False),
        sa.Column("target_date", sa.DateTime(), nullable=False),
        sa.Column("predicted_index", sa.Float(), nullable=False),
        sa.Column("lower_bound", sa.Float(), nullable=True),
        sa.Column("upper_bound", sa.Float(), nullable=True),
        sa.Column("confidence_label", sa.String(length=16), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_pollen_forecast_id", "pollen_forecast", ["id"], unique=False)
    op.create_index("ix_pollen_forecast_pollen_type", "pollen_forecast", ["pollen_type"], unique=False)
    op.create_index("ix_pollen_forecast_region_code", "pollen_forecast", ["region_code"], unique=False)
    op.create_index("ix_pollen_forecast_horizon_days", "pollen_forecast", ["horizon_days"], unique=False)
    op.create_index("ix_pollen_forecast_forecast_date", "pollen_forecast", ["forecast_date"], unique=False)
    op.create_index("ix_pollen_forecast_target_date", "pollen_forecast", ["target_date"], unique=False)
    op.create_index("ix_pollen_forecast_created_at", "pollen_forecast", ["created_at"], unique=False)
    op.create_index(
        "idx_forecast_scope_target",
        "pollen_forecast",
        ["pollen_type", "region_code", "horizon_days", "target_date"],
        unique=False,
    )
    op.create_index(
        "idx_forecast_scope_created",
        "pollen_forecast",
        ["pollen_type", "region_code", "horizon_days", "created_at"],
        unique=False,
    )

    op.create_table(
        "weather_data",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("datum", sa.DateTime(), nullable=False),
        sa.Column("available_time", sa.DateTime(), nullable=True),
        sa.Column("city", sa.String(length=64), nullable=False),
        sa.Column("region_code", sa.String(length=2), nullable=True),
        sa.Column("temperatur", sa.Float(), nullable=True),
        sa.Column("gefuehlte_temperatur", sa.Float(), nullable=True),
        sa.Column("luftfeuchtigkeit", sa.Float(), nullable=True),
        sa.Column("luftdruck", sa.Float(), nullable=True),
        sa.Column("wind_geschwindigkeit", sa.Float(), nullable=True),
        sa.Column("wolken", sa.Float(), nullable=True),
        sa.Column("niederschlag_wahrscheinlichkeit", sa.Float(), nullable=True),
        sa.Column("regen_mm", sa.Float(), nullable=True),
        sa.Column("taupunkt", sa.Float(), nullable=True),
        sa.Column("data_type", sa.String(length=32), nullable=False, server_default="CURRENT"),
        sa.Column("forecast_run_timestamp", sa.DateTime(), nullable=True),
        sa.Column("forecast_run_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_weather_data_id", "weather_data", ["id"], unique=False)
    op.create_index("ix_weather_data_datum", "weather_data", ["datum"], unique=False)
    op.create_index("ix_weather_data_available_time", "weather_data", ["available_time"], unique=False)
    op.create_index("ix_weather_data_region_code", "weather_data", ["region_code"], unique=False)
    op.create_index(
        "ix_weather_data_forecast_run_timestamp",
        "weather_data",
        ["forecast_run_timestamp"],
        unique=False,
    )
    op.create_index("ix_weather_data_forecast_run_id", "weather_data", ["forecast_run_id"], unique=False)
    op.create_index("idx_weather_date_city", "weather_data", ["datum", "city"], unique=False)
    op.create_index("idx_weather_region_date", "weather_data", ["region_code", "datum"], unique=False)
    op.create_index("idx_weather_data_type", "weather_data", ["data_type"], unique=False)

    op.create_table(
        "school_holidays",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("bundesland", sa.String(length=2), nullable=False),
        sa.Column("ferien_typ", sa.String(length=32), nullable=False),
        sa.Column("start_datum", sa.DateTime(), nullable=False),
        sa.Column("end_datum", sa.DateTime(), nullable=False),
        sa.Column("jahr", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_school_holidays_id", "school_holidays", ["id"], unique=False)
    op.create_index("ix_school_holidays_bundesland", "school_holidays", ["bundesland"], unique=False)
    op.create_index("ix_school_holidays_jahr", "school_holidays", ["jahr"], unique=False)
    op.create_index("idx_holidays_dates", "school_holidays", ["start_datum", "end_datum"], unique=False)
    op.create_index(
        "idx_holidays_bundesland_jahr", "school_holidays", ["bundesland", "jahr"], unique=False
    )

    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="success"),
        sa.Column("pollen_type", sa.String(length=32), nullable=False),
        sa.Column("region_code", sa.String(length=2), nullable=True),
        sa.Column("horizon_days", sa.Integer(), nullable=False, server_default="7"),
        sa.Column("min_train_points", sa.Integer(), nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("baseline_metrics", sa.JSON(), nullable=True),
        sa.Column("improvement_vs_baselines", sa.JSON(), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("chart_points", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_backtest_runs_id", "backtest_runs", ["id"], unique=False)
    op.create_index("ix_backtest_runs_run_id", "backtest_runs", ["run_id"], unique=True)
    op.create_index("ix_backtest_runs_status", "backtest_runs", ["status"], unique=False)
    op.create_index("ix_backtest_runs_pollen_type", "backtest_runs", ["pollen_type"], unique=False)
    op.create_index("ix_backtest_runs_region_code", "backtest_runs", ["region_code"], unique=False)
    op.create_index("ix_backtest_runs_created_at", "backtest_runs", ["created_at"], unique=False)
    op.create_index(
        "idx_backtest_scope_created",
        "backtest_runs",
        ["pollen_type", "region_code", "horizon_days", "created_at"],
        unique=False,
    )

    op.create_table(
        "backtest_points",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("date", sa.DateTime(), nullable=False),
        sa.Column("region_code", sa.String(length=2), nullable=True),
        sa.Column("real_value", sa.Float(), nullable=True),
        sa.Column("predicted_value", sa.Float(), nullable=True),
        sa.Column("lower_bound", sa.Float(), nullable=True),
        sa.Column("upper_bound", sa.Float(), nullable=True),
        sa.Column("baseline_persistence", sa.Float(), nullable=True),
        sa.Column("baseline_seasonal", sa.Float(), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["backtest_runs.run_id"]),
    )
    op.create_index("ix_backtest_points_id", "backtest_points", ["id"], unique=False)
    op.create_index("ix_backtest_points_run_id", "backtest_points", ["run_id"], unique=False)
    op.create_index("ix_backtest_points_date", "backtest_points", ["date"], unique=False)
    op.create_index(
        "ix_backtest_points_region_code", "backtest_points", ["region_code"], unique=False
    )
    op.create_index(
        "idx_backtest_points_run_date", "backtest_points", ["run_id", "date"], unique=False
    )

    op.create_table(
        "forecast_accuracy_log",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("computed_at", sa.DateTime(), nullable=False),
        sa.Column("pollen_type", sa.String(length=32), nullable=False),
        sa.Column("region_code", sa.String(length=2), nullable=True),
        sa.Column("horizon_days", sa.Integer(), nullable=False, server_default="7"),
        sa.Column("window_days", sa.Integer(), nullable=False, server_default="14"),
        sa.Column("samples", sa.Integer(), nullable=False),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("wis", sa.Float(), nullable=True),
        sa.Column("correlation", sa.Float(), nullable=True),
        sa.Column("drift_detected", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("details", sa.JSON(), nullable=True),
    )
    op.create_index("ix_forecast_accuracy_log_id", "forecast_accuracy_log", ["id"], unique=False)
    op.create_index(
        "ix_forecast_accuracy_log_computed_at", "forecast_accuracy_log", ["computed_at"], unique=False
    )
    op.create_index(
        "ix_forecast_accuracy_log_pollen_type", "forecast_accuracy_log", ["pollen_type"], unique=False
    )
    op.create_index(
        "ix_forecast_accuracy_log_region_code", "forecast_accuracy_log", ["region_code"], unique=False
    )
    op.create_index(
        "idx_accuracy_scope_computed",
        "forecast_accuracy_log",
        ["pollen_type", "region_code", "horizon_days", "computed_at"],
        unique=False,
    )

    op.create_table(
        "source_nowcast_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.String(length=32), nullable=False),
        sa.Column("signal_id", sa.String(length=64), nullable=False),
        sa.Column("region_code", sa.String(length=2), nullable=True),
        sa.Column("reference_date", sa.DateTime(), nullable=False),
        sa.Column("effective_available_time", sa.DateTime(), nullable=False),
        sa.Column("raw_value", sa.Float(), nullable=False),
        sa.Column("snapshot_captured_at", sa.DateTime(), nullable=False),
        sa.Column("timing_provenance", sa.String(length=32), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_source_nowcast_snapshots_id", "source_nowcast_snapshots", ["id"], unique=False
    )
    op.create_index(
        "ix_source_nowcast_snapshots_source_id",
        "source_nowcast_snapshots",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        "ix_source_nowcast_snapshots_signal_id",
        "source_nowcast_snapshots",
        ["signal_id"],
        unique=False,
    )
    op.create_index(
        "ix_source_nowcast_snapshots_region_code",
        "source_nowcast_snapshots",
        ["region_code"],
        unique=False,
    )
    op.create_index(
        "ix_source_nowcast_snapshots_reference_date",
        "source_nowcast_snapshots",
        ["reference_date"],
        unique=False,
    )
    op.create_index(
        "ix_source_nowcast_snapshots_effective_available_time",
        "source_nowcast_snapshots",
        ["effective_available_time"],
        unique=False,
    )
    op.create_index(
        "ix_source_nowcast_snapshots_snapshot_captured_at",
        "source_nowcast_snapshots",
        ["snapshot_captured_at"],
        unique=False,
    )
    op.create_index(
        "idx_nowcast_source_ref",
        "source_nowcast_snapshots",
        ["source_id", "reference_date"],
        unique=False,
    )
    op.create_index(
        "idx_nowcast_signal_region",
        "source_nowcast_snapshots",
        ["signal_id", "region_code"],
        unique=False,
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False, server_default="user"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_users_id", "users", ["id"], unique=False)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("user", sa.String(length=255), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=True),
        sa.Column("entity_id", sa.String(length=64), nullable=True),
        sa.Column("old_value", sa.JSON(), nullable=True),
        sa.Column("new_value", sa.JSON(), nullable=True),
        sa.Column("reason", sa.String(length=255), nullable=True),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_audit_logs_id", "audit_logs", ["id"], unique=False)
    op.create_index("ix_audit_logs_timestamp", "audit_logs", ["timestamp"], unique=False)
    op.create_index("ix_audit_logs_user", "audit_logs", ["user"], unique=False)

    op.create_table(
        "upload_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("upload_type", sa.String(length=32), nullable=False),
        sa.Column("file_format", sa.String(length=8), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("date_range_start", sa.DateTime(), nullable=True),
        sa.Column("date_range_end", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="success"),
        sa.Column("error_message", sa.String(length=512), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_upload_history_id", "upload_history", ["id"], unique=False)
    op.create_index("ix_upload_history_created_at", "upload_history", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_table("upload_history")
    op.drop_table("audit_logs")
    op.drop_table("users")
    op.drop_table("source_nowcast_snapshots")
    op.drop_table("forecast_accuracy_log")
    op.drop_table("backtest_points")
    op.drop_table("backtest_runs")
    op.drop_table("school_holidays")
    op.drop_table("weather_data")
    op.drop_table("pollen_forecast")
    op.drop_table("pollen_data")
