"""Backtest 'broker' — fully-deterministic order simulation against the cost model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from squeeze_hunter.backtest.cost_model import Fill, StockCostModel


@dataclass
class Lot:
    qty: int
    avg_price: float
    opened_at: datetime


@dataclass
class SimulatorBroker:
    initial_cash: float
    cost_model: StockCostModel
    cash: float = field(init=False)
    equity: float = field(init=False)
    positions: dict[str, Lot] = field(default_factory=dict)
    realized: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def __post_init__(self: SimulatorBroker) -> None:
        self.cash = self.initial_cash
        self.equity = self.initial_cash

    async def submit_buy(
        self: SimulatorBroker,
        ticker: str,
        qty: int,
        reference_price: float,
        ts: datetime,
        is_open_5min: bool = False,
    ) -> Fill:
        fill = self.cost_model.simulate_buy(reference_price, qty, is_open_5min=is_open_5min)
        cost = fill.fill_price * qty + fill.commission_usd
        self.cash -= cost
        existing = self.positions.get(ticker)
        if existing:
            new_qty = existing.qty + qty
            new_avg = (existing.avg_price * existing.qty + fill.fill_price * qty) / new_qty
            self.positions[ticker] = Lot(new_qty, new_avg, existing.opened_at)
        else:
            self.positions[ticker] = Lot(qty, fill.fill_price, ts)
        return fill

    async def submit_sell(
        self: SimulatorBroker,
        ticker: str,
        qty: int,
        reference_price: float,
        ts: datetime,
        is_open_5min: bool = False,
    ) -> Fill:
        fill = self.cost_model.simulate_sell(reference_price, qty, is_open_5min=is_open_5min)
        proceeds = fill.fill_price * qty - fill.commission_usd
        self.cash += proceeds
        existing = self.positions.get(ticker)
        if existing is None or existing.qty < qty:
            raise ValueError(f"insufficient position to sell {qty} of {ticker}")
        realized_pl = (fill.fill_price - existing.avg_price) * qty - fill.commission_usd
        self.realized[ticker] += realized_pl
        new_qty = existing.qty - qty
        if new_qty == 0:
            del self.positions[ticker]
        else:
            self.positions[ticker] = Lot(new_qty, existing.avg_price, existing.opened_at)
        return fill

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
