from datetime import UTC, datetime

import pytest

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.simulator import SimulatorBroker


@pytest.mark.asyncio
async def test_buy_then_sell_realizes_pnl() -> None:
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    await broker.submit_buy(
        ticker="GME",
        qty=100,
        limit_price=100.0,
        ts=datetime(2024, 5, 13, 14, 0, tzinfo=UTC),
    )
    assert broker.position_qty("GME") == 100
    await broker.submit_sell(
        ticker="GME",
        qty=100,
        limit_price=120.0,
        ts=datetime(2024, 5, 17, 14, 0, tzinfo=UTC),
    )
    assert broker.position_qty("GME") == 0
    # Realized PnL = (sell - buy) * 100 - 2*commission
    assert broker.realized_pnl("GME") > 1_500.0
    assert broker.cash > 100_000.0


@pytest.mark.asyncio
async def test_mark_to_market_updates_equity() -> None:
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    await broker.submit_buy(
        ticker="GME",
        qty=100,
        limit_price=100.0,
        ts=datetime(2024, 5, 13, tzinfo=UTC),
    )
    broker.mark_to_market({"GME": 130.0}, ts=datetime(2024, 5, 14, tzinfo=UTC))
    assert broker.equity > 100_000.0 + 100 * 30 - 100  # net of slippage and commission


@pytest.mark.asyncio
async def test_fetch_quote_returns_latest_mark_not_entry() -> None:
    """R7.C6 regression: SimulatorBroker.fetch_quote must serve the latest
    mark_to_market price, NOT the position's avg entry. Otherwise lifecycle
    stop evaluation in sim mode always sees pnl_pct = 0 and no stop fires.
    """
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    await broker.submit_buy(
        ticker="GME",
        qty=100,
        limit_price=100.0,
        ts=datetime(2026, 5, 11, 14, 0, tzinfo=UTC),
    )
    # Before any MTM, fetch_quote falls back to the entry-fill price.
    q0 = await broker.fetch_quote("GME")
    assert q0.last > 0  # acceptable bootstrap behavior

    # After MTM, fetch_quote must use the current mark.
    broker.mark_to_market({"GME": 70.0}, ts=datetime(2026, 5, 12, 14, 0, tzinfo=UTC))
    q1 = await broker.fetch_quote("GME")
    assert q1.last == pytest.approx(70.0)
    assert q1.bid == pytest.approx(70.0)
    assert q1.ask == pytest.approx(70.0)
