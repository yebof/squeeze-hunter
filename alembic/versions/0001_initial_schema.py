"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-10
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "universe",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("market_cap_usd", sa.Float, nullable=False),
        sa.Column("close_price", sa.Float, nullable=False),
        sa.Column("included", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("exclusion_reason", sa.String(64), nullable=True),
        sa.UniqueConstraint("ticker", "as_of", name="uq_universe_ticker_date"),
    )
    op.create_index("ix_universe_ticker", "universe", ["ticker"])
    op.create_index("ix_universe_as_of", "universe", ["as_of"])

    op.create_table(
        "signals_daily",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("factor_name", sa.String(64), nullable=False),
        sa.Column("raw_value", sa.Float, nullable=False),
        sa.Column("z_score", sa.Float, nullable=False),
        sa.UniqueConstraint("as_of", "ticker", "factor_name", name="uq_signal_unique"),
    )
    op.create_index("ix_signals_as_of", "signals_daily", ["as_of"])
    op.create_index("ix_signals_ticker", "signals_daily", ["ticker"])

    op.create_table(
        "setup_classifications",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("setup_type", sa.String(16), nullable=False),
        sa.Column("rank_in_universe", sa.Integer, nullable=False),
        sa.UniqueConstraint("as_of", "ticker", name="uq_setup_unique"),
    )

    op.create_table(
        "positions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("instrument", sa.String(16), nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False),
        sa.Column("avg_price", sa.Float, nullable=False),
        sa.Column("setup_type", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("realized_pnl", sa.Float, nullable=False, server_default="0"),
        sa.Column("option_strike", sa.Float, nullable=True),
        sa.Column("option_expiry", sa.Date, nullable=True),
    )

    op.create_table(
        "orders",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("client_order_id", sa.String(64), nullable=False, unique=True),
        sa.Column("broker_order_id", sa.String(64), nullable=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("instrument", sa.String(16), nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False),
        sa.Column("limit_price", sa.Float, nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("avg_fill_price", sa.Float, nullable=True),
        sa.Column("position_id", sa.Integer, sa.ForeignKey("positions.id"), nullable=True),
    )

    op.create_table(
        "pnl_daily",
        sa.Column("as_of", sa.Date, primary_key=True),
        sa.Column("equity_usd", sa.Float, nullable=False),
        sa.Column("cash_usd", sa.Float, nullable=False),
        sa.Column("gross_exposure_pct", sa.Float, nullable=False),
        sa.Column("daily_pnl_usd", sa.Float, nullable=False),
        sa.Column("drawdown_pct", sa.Float, nullable=False),
    )

    op.create_table(
        "kill_switch_events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason", sa.String(64), nullable=False),
        sa.Column("details_json", sa.JSON, nullable=False),
        sa.Column("cleared_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    for t in (
        "kill_switch_events",
        "pnl_daily",
        "orders",
        "positions",
        "setup_classifications",
        "signals_daily",
        "universe",
    ):
        op.drop_table(t)
