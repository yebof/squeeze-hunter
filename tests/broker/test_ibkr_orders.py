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
