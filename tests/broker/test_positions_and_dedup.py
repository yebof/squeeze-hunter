"""P2 — get_positions on every broker; IBKR client order refs are deduped."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.ibkr import IBKRBroker
from squeeze_hunter.broker.simulator import SimulatorBroker

_TS = datetime(2026, 5, 14, 14, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_simulator_get_positions_lists_lots() -> None:
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    await broker.submit_buy("GME", 100, 18.0, _TS)
    await broker.submit_buy("AMC", 10, 5.0, _TS)
    snaps = {p.ticker: p for p in await broker.get_positions()}
    assert snaps["GME"].qty == 100
    assert snaps["GME"].avg_cost == pytest.approx(broker.positions["GME"].avg_price)
    assert set(snaps) == {"GME", "AMC"}


def _row(symbol: str, account: str, qty: int, avg_cost: float) -> MagicMock:
    pos = MagicMock()
    pos.contract.symbol = symbol
    pos.account = account
    pos.position = qty
    pos.avgCost = avg_cost
    return pos


@pytest.mark.asyncio
async def test_ibkr_get_positions_filters_account_and_skips_flat_rows() -> None:
    broker = IBKRBroker(client_id=1, account="DU111")
    fake = MagicMock()
    fake.reqPositionsAsync = AsyncMock()
    fake.positions = MagicMock(
        return_value=[
            _row("GME", "DU111", 100, 18.0),
            _row("GME", "U222", 50, 17.0),
            _row("AMC", "DU111", 0, 5.0),
        ]
    )
    broker._ib = fake
    snaps = await broker.get_positions()
    assert [(p.ticker, p.qty, p.avg_cost) for p in snaps] == [("GME", 100, 18.0)]


def _fake_ib_with_orders() -> tuple[MagicMock, list]:
    fake = MagicMock()
    fake.qualifyContractsAsync = AsyncMock()
    placed: list = []

    def place(contract, order):
        placed.append(order)
        trade = SimpleNamespace(
            order=SimpleNamespace(orderId=len(placed), orderRef=order.orderRef),
            orderStatus=SimpleNamespace(status="PendingSubmit", filled=0, avgFillPrice=0.0),
            contract=contract,
        )
        fake._open.append(trade)
        return trade

    fake._open = []
    fake.placeOrder = MagicMock(side_effect=place)
    fake.openTrades = MagicMock(side_effect=lambda: list(fake._open))
    return fake, placed


@pytest.mark.asyncio
async def test_ibkr_dedupes_a_resubmitted_client_order_ref() -> None:
    """The same logical order (ticker, side, qty, timestamp) submitted twice —
    e.g. a retry after a timeout whose first attempt actually went through —
    must place exactly one broker order and return the existing one."""
    broker = IBKRBroker(client_id=1)
    fake, placed = _fake_ib_with_orders()
    broker._ib = fake
    first = await broker.submit_sell("GME", 100, 24.4, _TS)
    second = await broker.submit_sell("GME", 100, 24.4, _TS)
    assert len(placed) == 1
    assert placed[0].orderRef.startswith("sh-sell-GME-100-")
    assert second.broker_order_id == first.broker_order_id
    # A different logical order is not deduped.
    await broker.submit_sell("GME", 100, 24.4, _TS.replace(minute=1))
    assert len(placed) == 2
