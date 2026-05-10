"""Backtest 'broker' — fully-deterministic order simulation against the cost model."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from squeeze_hunter.backtest.cost_model import Fill, StockCostModel
from squeeze_hunter.broker.base import BrokerHealth, BrokerOrder, Quote
from squeeze_hunter.logging_setup import get_logger

log = get_logger("broker.simulator")


@dataclass
class Lot:
    qty: int
    avg_price: float
    opened_at: datetime


@dataclass
class SimulatorBroker:
    initial_cash: float
    cost_model: StockCostModel
    name: str = "simulator"
    cash: float = field(init=False)
    equity: float = field(init=False)
    positions: dict[str, Lot] = field(default_factory=dict)
    realized: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def __post_init__(self: SimulatorBroker) -> None:
        self.cash = self.initial_cash
        self.equity = self.initial_cash

    async def connect(self: SimulatorBroker) -> None:
        pass

    async def disconnect(self: SimulatorBroker) -> None:
        pass

    async def health(self: SimulatorBroker) -> BrokerHealth:
        return BrokerHealth(connected=True, last_ping_ms=0, account="simulator")

    async def fetch_quote(self: SimulatorBroker, ticker: str) -> Quote:
        lot = self.positions.get(ticker)
        if lot is not None:
            price = lot.avg_price
            return Quote(
                ticker=ticker,
                bid=price,
                ask=price,
                last=price,
                timestamp_ns=time.time_ns(),
            )
        return Quote(
            ticker=ticker,
            bid=0.0,
            ask=0.0,
            last=0.0,
            timestamp_ns=time.time_ns(),
        )

    async def submit_buy(
        self: SimulatorBroker,
        ticker: str,
        qty: int,
        limit_price: float | None,
        ts: datetime,
        is_open_5min: bool = False,
    ) -> BrokerOrder:
        reference_price = _resolve_price(limit_price, self.positions.get(ticker), ticker, "buy")
        fill: Fill = self.cost_model.simulate_buy(reference_price, qty, is_open_5min=is_open_5min)
        cost = fill.fill_price * qty + fill.commission_usd
        self.cash -= cost
        existing = self.positions.get(ticker)
        if existing:
            new_qty = existing.qty + qty
            new_avg = (existing.avg_price * existing.qty + fill.fill_price * qty) / new_qty
            self.positions[ticker] = Lot(new_qty, new_avg, existing.opened_at)
        else:
            self.positions[ticker] = Lot(qty, fill.fill_price, ts)
        order_id = f"sim-{ticker}-{ts.isoformat()}-buy-{uuid.uuid4().hex[:8]}"
        return BrokerOrder(
            broker_order_id=order_id,
            ticker=ticker,
            side="buy",
            qty=qty,
            limit_price=limit_price,
            status="filled",
            filled_qty=qty,
            avg_fill_price=fill.fill_price,
        )

    async def submit_sell(
        self: SimulatorBroker,
        ticker: str,
        qty: int,
        limit_price: float | None,
        ts: datetime,
        is_open_5min: bool = False,
    ) -> BrokerOrder:
        existing = self.positions.get(ticker)
        if existing is None or existing.qty < qty:
            raise ValueError(f"insufficient position to sell {qty} of {ticker}")
        reference_price = _resolve_price(limit_price, existing, ticker, "sell")
        fill: Fill = self.cost_model.simulate_sell(reference_price, qty, is_open_5min=is_open_5min)
        proceeds = fill.fill_price * qty - fill.commission_usd
        self.cash += proceeds
        realized_pl = (fill.fill_price - existing.avg_price) * qty - fill.commission_usd
        self.realized[ticker] += realized_pl
        new_qty = existing.qty - qty
        if new_qty == 0:
            del self.positions[ticker]
        else:
            self.positions[ticker] = Lot(new_qty, existing.avg_price, existing.opened_at)
        order_id = f"sim-{ticker}-{ts.isoformat()}-sell-{uuid.uuid4().hex[:8]}"
        return BrokerOrder(
            broker_order_id=order_id,
            ticker=ticker,
            side="sell",
            qty=qty,
            limit_price=limit_price,
            status="filled",
            filled_qty=qty,
            avg_fill_price=fill.fill_price,
        )

    async def cancel_order(self: SimulatorBroker, broker_order_id: str) -> bool:
        # Simulator fills immediately; no orders are ever open to cancel.
        return False

    async def get_open_orders(self: SimulatorBroker) -> list[BrokerOrder]:
        return []

    def position_qty(self: SimulatorBroker, ticker: str) -> int:
        lot = self.positions.get(ticker)
        return lot.qty if lot else 0

    def realized_pnl(self: SimulatorBroker, ticker: str) -> float:
        return self.realized.get(ticker, 0.0)

    def gross_exposure_pct(self: SimulatorBroker, marks: dict[str, float]) -> float:
        if self.equity <= 0:
            return 0.0
        notional = sum(marks.get(t, lot.avg_price) * lot.qty for t, lot in self.positions.items())
        return notional / self.equity

    def mark_to_market(self: SimulatorBroker, marks: dict[str, float], ts: datetime) -> None:
        notional = sum(marks.get(t, lot.avg_price) * lot.qty for t, lot in self.positions.items())
        self.equity = self.cash + notional


def _resolve_price(
    limit_price: float | None,
    lot: Lot | None,
    ticker: str,
    side: str,
) -> float:
    """Return a reference price for the cost model.

    Priority:
    1. limit_price if given (non-None)
    2. lot.avg_price if an existing position is known (market-order against open pos)
    3. 1.0 as a last-resort placeholder (logs a warning so this is detectable)
    """
    if limit_price is not None:
        return limit_price
    if lot is not None:
        return lot.avg_price
    log.warning(
        "market_order_no_reference",
        ticker=ticker,
        side=side,
        msg="no limit_price and no existing position; using placeholder $1.00",
    )
    return 1.0
