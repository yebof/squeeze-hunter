"""P3 — the OMS no longer assumes a slice fills on the submitting call."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from squeeze_hunter.broker.base import BrokerOrder, Quote
from squeeze_hunter.execution.oms import OrderManager
from squeeze_hunter.execution.slicing import build_twap_plan


class _AsyncFillBroker:
    """submit_* returns pending; the order fills on the N-th get_order poll."""

    def __init__(self, fill_after_polls: int | None) -> None:
        self.fill_after_polls = fill_after_polls
        self.polls: dict[str, int] = {}
        self.orders: dict[str, BrokerOrder] = {}
        self.cancelled: list[str] = []

    async def fetch_quote(self, ticker: str) -> Quote:
        return Quote(ticker=ticker, bid=20.0, ask=20.1, last=20.05, timestamp_ns=0)

    async def submit_buy(self, ticker, qty, limit_price, ts) -> BrokerOrder:
        oid = f"o-{len(self.orders) + 1}"
        self.orders[oid] = BrokerOrder(
            broker_order_id=oid,
            ticker=ticker,
            side="buy",
            qty=qty,
            limit_price=limit_price,
            status="pending",
        )
        return self.orders[oid]

    async def get_order(self, order_id: str) -> BrokerOrder | None:
        self.polls[order_id] = self.polls.get(order_id, 0) + 1
        o = self.orders[order_id]
        if self.fill_after_polls is not None and self.polls[order_id] >= self.fill_after_polls:
            o = BrokerOrder(
                broker_order_id=order_id,
                ticker=o.ticker,
                side=o.side,
                qty=o.qty,
                limit_price=o.limit_price,
                status="filled",
                filled_qty=o.qty,
                avg_fill_price=o.limit_price,
            )
            self.orders[order_id] = o
        return o

    async def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        o = self.orders[order_id]
        self.orders[order_id] = BrokerOrder(
            broker_order_id=order_id,
            ticker=o.ticker,
            side=o.side,
            qty=o.qty,
            limit_price=o.limit_price,
            status="cancelled",
            filled_qty=o.filled_qty,
            avg_fill_price=o.avg_fill_price,
        )
        return True

    async def get_open_orders(self) -> list[BrokerOrder]:
        return [o for o in self.orders.values() if o.status in {"pending", "routed", "partial"}]


def _plan(open_at: datetime):
    return build_twap_plan(300, 20.0, open_at, ticker="GME", side="buy", n_slices=3)


@pytest.mark.asyncio
async def test_oms_counts_fills_that_arrive_after_submission() -> None:
    broker = _AsyncFillBroker(fill_after_polls=2)
    open_at = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
    oms = OrderManager(broker=broker, clock=lambda: open_at + timedelta(hours=1))  # type: ignore[arg-type]
    result = await oms.execute(
        _plan(open_at), max_wall_seconds=0, poll_interval_s=0, slice_fill_wait_s=5
    )
    assert result.filled_qty == 300
    assert result.unfilled_qty == 0
    assert result.avg_fill_price > 0
    assert broker.cancelled == []


@pytest.mark.asyncio
async def test_oms_cancels_a_slice_that_never_fills_and_reports_it_unfilled() -> None:
    broker = _AsyncFillBroker(fill_after_polls=None)
    open_at = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
    oms = OrderManager(broker=broker, clock=lambda: open_at + timedelta(hours=1))  # type: ignore[arg-type]
    result = await oms.execute(
        _plan(open_at), max_wall_seconds=0, poll_interval_s=0, slice_fill_wait_s=0.0
    )
    assert result.filled_qty == 0
    assert result.unfilled_qty == 300
    assert len(broker.cancelled) == 3
