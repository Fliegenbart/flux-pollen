"""Add outcome_observations table (Phase 5a — customer pilot uploads).

Holds commercial outcome data supplied by customers (sell-out, media,
search). Keyed on (brand, product, region, metric, week window,
source_label) so the same customer can upload the same window twice
without duplicating rows.

Revision ID: 0003_outcome_observations
Revises: 0002_pollen_observations
Create Date: 2026-04-18

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_outcome_observations"
down_revision = "0002_pollen_observations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "outcome_observations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("brand", sa.String(length=64), nullable=False, server_default="hexal"),
        sa.Column("product", sa.String(length=64), nullable=False),
        sa.Column("region_code", sa.String(length=2), nullable=False),
        sa.Column("window_start", sa.DateTime(), nullable=False),
        sa.Column("window_end", sa.DateTime(), nullable=False),
        sa.Column("metric_name", sa.String(length=64), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("metric_unit", sa.String(length=32), nullable=True),
        sa.Column("source_label", sa.String(length=64), nullable=False, server_default="customer_upload"),
        sa.Column("channel", sa.String(length=64), nullable=True),
        sa.Column("campaign_id", sa.String(length=64), nullable=True),
        sa.Column("holdout_group", sa.String(length=32), nullable=True),
        sa.Column("confidence_hint", sa.Float(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_outcome_observations_id", "outcome_observations", ["id"], unique=False)
    op.create_index("ix_outcome_observations_brand", "outcome_observations", ["brand"], unique=False)
    op.create_index("ix_outcome_observations_product", "outcome_observations", ["product"], unique=False)
    op.create_index("ix_outcome_observations_region_code", "outcome_observations", ["region_code"], unique=False)
    op.create_index("ix_outcome_observations_window_start", "outcome_observations", ["window_start"], unique=False)
    op.create_index("ix_outcome_observations_window_end", "outcome_observations", ["window_end"], unique=False)
    op.create_index("ix_outcome_observations_metric_name", "outcome_observations", ["metric_name"], unique=False)
    op.create_index("ix_outcome_observations_source_label", "outcome_observations", ["source_label"], unique=False)
    op.create_index("ix_outcome_observations_channel", "outcome_observations", ["channel"], unique=False)
    op.create_index("ix_outcome_observations_campaign_id", "outcome_observations", ["campaign_id"], unique=False)
    op.create_index("ix_outcome_observations_created_at", "outcome_observations", ["created_at"], unique=False)
    op.create_index("idx_outcome_brand_window", "outcome_observations", ["brand", "window_start"], unique=False)
    op.create_index("idx_outcome_metric_window", "outcome_observations", ["metric_name", "window_start"], unique=False)
    op.create_index("idx_outcome_region_product", "outcome_observations", ["region_code", "product"], unique=False)
    op.create_unique_constraint(
        "uq_outcome_observation",
        "outcome_observations",
        ["window_start", "window_end", "brand", "product", "region_code", "metric_name", "source_label"],
    )


def downgrade() -> None:
    op.drop_table("outcome_observations")
