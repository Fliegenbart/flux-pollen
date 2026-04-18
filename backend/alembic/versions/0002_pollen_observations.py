"""Add pollen_observations table (station-level concentration data).

Primary backing store for ePIN Bayern ingestion and any future
station-level monitoring feeds. Parallel to pollen_data, which holds
the DWD regional index (0–3). Keeping the two tables separate avoids a
lossy unit conflation; the feature builder decides which source to
consume per (region, pollen_type, horizon).

Revision ID: 0002_pollen_observations
Revises: 0001_initial_schema
Create Date: 2026-04-18

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_pollen_observations"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pollen_observations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("station_id", sa.String(length=16), nullable=False),
        sa.Column("station_name", sa.String(length=64), nullable=True),
        sa.Column("region_code", sa.String(length=2), nullable=False),
        sa.Column("pollen_type", sa.String(length=32), nullable=False),
        sa.Column("from_time", sa.DateTime(), nullable=False),
        sa.Column("to_time", sa.DateTime(), nullable=False),
        sa.Column("concentration", sa.Float(), nullable=False),
        sa.Column("algorithm", sa.String(length=32), nullable=True),
        sa.Column("source_network", sa.String(length=16), nullable=False, server_default="ePIN"),
        sa.Column("available_time", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_pollen_observations_id", "pollen_observations", ["id"], unique=False)
    op.create_index(
        "ix_pollen_observations_station_id", "pollen_observations", ["station_id"], unique=False
    )
    op.create_index(
        "ix_pollen_observations_region_code", "pollen_observations", ["region_code"], unique=False
    )
    op.create_index(
        "ix_pollen_observations_pollen_type", "pollen_observations", ["pollen_type"], unique=False
    )
    op.create_index(
        "ix_pollen_observations_from_time", "pollen_observations", ["from_time"], unique=False
    )
    op.create_index(
        "ix_pollen_observations_created_at", "pollen_observations", ["created_at"], unique=False
    )
    op.create_index(
        "idx_pollen_obs_station_type_from",
        "pollen_observations",
        ["station_id", "pollen_type", "from_time"],
        unique=False,
    )
    op.create_index(
        "idx_pollen_obs_region_type_from",
        "pollen_observations",
        ["region_code", "pollen_type", "from_time"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_pollen_obs_station_type_window",
        "pollen_observations",
        ["station_id", "pollen_type", "from_time"],
    )


def downgrade() -> None:
    op.drop_table("pollen_observations")
