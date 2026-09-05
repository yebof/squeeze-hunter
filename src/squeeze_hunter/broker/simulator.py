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
    # R7.C6: last per-ticker mark fed in via mark_to_market(). fetch_quote
    # uses this so lifecycle stop evaluation sees the CURRENT bar's price,
    # not the stale `lot.avg_price` (entry). Without this, sim-mode stops
    # never fire because pnl_pct = (entry - entry) / entry = 0 forever.
    last_marks: dict[str, float] = field(default_factory=dict)

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
        # R7.C6: prefer the most recent mark-to-market price (real current
        # bar) over lot.avg_price (entry, stale). Without this, lifecycle
        # daemon's stop evaluation in sim mode always sees pnl_pct = 0 and
        # no stop ever fires.
        price = self.last_marks.get(ticker)
        if price is None or price <= 0:
            lot = self.positions.get(ticker)
            price = lot.avg_price if lot is not None else 0.0
        if price > 0:
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
            # R9.10: surface the commission so the backtest runner can
            # compute net realized P&L for Kelly avg_payoff accuracy.
            commission_usd=fill.commission_usd,
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
            # R8.Q-I8: drop the per-ticker mark when the position closes so
            # last_marks doesn't grow unbounded across long backtests with a
            # rotating universe.
            self.last_marks.pop(ticker, None)
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
            # R9.10: see submit_buy.
            commission_usd=fill.commission_usd,
        )

    async def cancel_order(self: SimulatorBroker, broker_order_id: str) -> bool:
        # Simulator fills immediately; no orders are ever open to cancel.
        return False

    async def get_open_orders(self: SimulatorBroker) -> list[BrokerOrder]:
        return []

    async def get_equity_usd(self: SimulatorBroker) -> float | None:
        """R4.1: return the simulator's tracked equity (cash + last mark-to-market)."""
        return self.equity

    def position_qty(self: SimulatorBroker, ticker: str) -> int:
        lot = self.positions.get(ticker)
        return lot.qty if lot else 0

    async def get_position_qty(self: SimulatorBroker, ticker: str) -> int:
        """CDX-P1-3: IBroker Protocol position query. The simulator fills
        synchronously so its local lot IS the authoritative position."""
        return self.position_qty(ticker)

    def realized_pnl(self: SimulatorBroker, ticker: str) -> float:
        return self.realized.get(ticker, 0.0)

    def gross_exposure_pct(self: SimulatorBroker, marks: dict[str, float]) -> float:
        if self.equity <= 0:
            return 0.0
        # Round-13: a ticker with no fresh mark keeps its LAST mark, not its
        # entry price — falling back to cost erased accrued P&L from equity,
        # gross exposure and the drawdown killswitch input on any no-bar day.
        notional = sum(
            marks.get(t, self.last_marks.get(t, lot.avg_price)) * lot.qty
            for t, lot in self.positions.items()
        )
        return notional / self.equity

    def mark_to_market(self: SimulatorBroker, marks: dict[str, float], ts: datetime) -> None:
        # R7.C6: remember the latest per-ticker marks so fetch_quote can serve
        # the current price (not entry) to lifecycle/oms callers.
        for t, px in marks.items():
            if px > 0:
                self.last_marks[t] = px
        # Round-13: a ticker with no fresh mark keeps its LAST mark, not its
        # entry price — falling back to cost erased accrued P&L from equity,
        # gross exposure and the drawdown killswitch input on any no-bar day.
        notional = sum(
            marks.get(t, self.last_marks.get(t, lot.avg_price)) * lot.qty
            for t, lot in self.positions.items()
        )
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
