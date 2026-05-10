"""SQLAlchemy ORM models. One row = one fact at one time."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UniverseRow(Base):
    __tablename__ = "universe"
    __table_args__ = (UniqueConstraint("ticker", "as_of", name="uq_universe_ticker_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    market_cap_usd: Mapped[float] = mapped_column(Float)
    close_price: Mapped[float] = mapped_column(Float)
    included: Mapped[bool] = mapped_column(Boolean, default=True)
    exclusion_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)


class SignalDaily(Base):
    __tablename__ = "signals_daily"
    __table_args__ = (UniqueConstraint("as_of", "ticker", "factor_name", name="uq_signal_unique"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    factor_name: Mapped[str] = mapped_column(String(64))
    raw_value: Mapped[float] = mapped_column(Float)
    z_score: Mapped[float] = mapped_column(Float)


class SetupClassification(Base):
    __tablename__ = "setup_classifications"
    __table_args__ = (UniqueConstraint("as_of", "ticker", name="uq_setup_unique"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    score: Mapped[float] = mapped_column(Float)
    setup_type: Mapped[str] = mapped_column(String(16))  # CAR, GME, Mixed, Weak
    rank_in_universe: Mapped[int] = mapped_column(Integer)


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    side: Mapped[str] = mapped_column(String(8))  # long
    instrument: Mapped[str] = mapped_column(String(16))  # stock, call, put
    quantity: Mapped[int] = mapped_column(Integer)
    avg_price: Mapped[float] = mapped_column(Float)
    setup_type: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="open")
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    option_strike: Mapped[float | None] = mapped_column(Float, nullable=True)
    option_expiry: Mapped[date | None] = mapped_column(Date, nullable=True)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(8))  # buy, sell
    instrument: Mapped[str] = mapped_column(String(16))
    quantity: Mapped[int] = mapped_column(Integer)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    avg_fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_id: Mapped[int | None] = mapped_column(ForeignKey("positions.id"), nullable=True)


class PnLDaily(Base):
    __tablename__ = "pnl_daily"

    as_of: Mapped[date] = mapped_column(Date, primary_key=True)
    equity_usd: Mapped[float] = mapped_column(Float)
    cash_usd: Mapped[float] = mapped_column(Float)
    gross_exposure_pct: Mapped[float] = mapped_column(Float)
    daily_pnl_usd: Mapped[float] = mapped_column(Float)
    drawdown_pct: Mapped[float] = mapped_column(Float)


class KillSwitchEvent(Base):
    __tablename__ = "kill_switch_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reason: Mapped[str] = mapped_column(String(64))
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
