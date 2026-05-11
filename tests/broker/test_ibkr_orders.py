from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from squeeze_hunter.broker.base import BrokerOrder
from squeeze_hunter.broker.ibkr import IBKRBroker


@pytest.mark.asyncio
async def test_submit_buy_returns_pending_order() -> None:
    broker = IBKRBroker(client_id=99)
    fake_ib = MagicMock()
    fake_trade = MagicMock()
    fake_trade.order.orderId = 1234
    fake_trade.orderStatus.status = "PreSubmitted"
    fake_ib.placeOrder = MagicMock(return_value=fake_trade)
    fake_ib.qualifyContractsAsync = AsyncMock()
    broker._ib = fake_ib

    order = await broker.submit_buy(
        ticker="GME",
        qty=100,
        limit_price=18.5,
        ts=datetime(2026, 5, 14, 13, 35, tzinfo=UTC),
    )
    assert isinstance(order, BrokerOrder)
    assert order.broker_order_id == "1234"
    assert order.status == "pending"
    assert order.side == "buy"
    assert order.qty == 100


@pytest.mark.asyncio
async def test_connect_subscribes_to_account_updates() -> None:
    """R5.C1 regression: IBKRBroker.connect() must call reqAccountUpdates so
    accountValues() is populated and get_equity_usd works in real IBKR mode.

    Before the fix: get_equity_usd looped over an empty list and returned
    None forever. The drawdown/3-day-loss killswitch arms were dead in
    paper/live mode despite the R4.1 wiring claiming to fix them.
    """
    broker = IBKRBroker(client_id=99)
    fake_ib = MagicMock()
    fake_ib.connectAsync = AsyncMock()
    fake_ib.client.serverVersion = MagicMock(return_value=176)
    fake_ib.reqAccountUpdates = MagicMock()
    broker._ib = fake_ib

    await broker.connect()

    # R6.I2: assert the exact argument, not just that the method was called.
    # ib-async's reqAccountUpdates(account: str) — calling it IS the
    # subscribe call. The default account is "" (IBKR uses primary account).
    assert fake_ib.reqAccountUpdates.called
    fake_ib.reqAccountUpdates.assert_called_with("")


@pytest.mark.asyncio
async def test_get_equity_usd_returns_net_liquidation_usd() -> None:
    """R5.C1: get_equity_usd parses NetLiquidation/USD from accountValues."""
    broker = IBKRBroker(client_id=99)
    fake_ib = MagicMock()

    # Simulate ib-async AccountValue objects (just need tag/currency/value attrs)
    av_nav = MagicMock()
    av_nav.tag = "NetLiquidation"
    av_nav.currency = "USD"
    av_nav.value = "87500.00"
    av_other = MagicMock()
    av_other.tag = "CashBalance"
    av_other.currency = "USD"
    av_other.value = "10000"

    fake_ib.accountValues = MagicMock(return_value=[av_other, av_nav])
    fake_ib.reqAccountUpdates = MagicMock()
    broker._ib = fake_ib

    equity = await broker.get_equity_usd()
    assert equity == 87500.0


@pytest.mark.asyncio
async def test_get_equity_usd_returns_none_when_no_nav() -> None:
    """No NetLiquidation entry yet (just-connected) → None, not 0.
    None tells the runtime to skip recording so the killswitch isn't tripped
    on a phantom zero.
    """
    broker = IBKRBroker(client_id=99)
    fake_ib = MagicMock()
    fake_ib.accountValues = MagicMock(return_value=[])
    fake_ib.reqAccountUpdates = MagicMock()
    broker._ib = fake_ib

    equity = await broker.get_equity_usd()
    assert equity is None


@pytest.mark.asyncio
async def test_submit_sell_uses_limit_when_provided() -> None:
    broker = IBKRBroker(client_id=99)
    fake_ib = MagicMock()
    fake_trade = MagicMock()
    fake_trade.order.orderId = 5678
    fake_trade.orderStatus.status = "PreSubmitted"
    captured = {}

    def _capture(contract, order):
        captured["limit"] = order.lmtPrice
        captured["action"] = order.action
        return fake_trade

    fake_ib.placeOrder = _capture
    fake_ib.qualifyContractsAsync = AsyncMock()
    broker._ib = fake_ib

    await broker.submit_sell(
        ticker="GME",
        qty=50,
        limit_price=20.0,
        ts=datetime(2026, 5, 14, 13, 35, tzinfo=UTC),
    )
    assert captured["limit"] == 20.0
    assert captured["action"] == "SELL"
