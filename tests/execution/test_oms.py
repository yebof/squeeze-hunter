from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from squeeze_hunter.broker.base import BrokerOrder
from squeeze_hunter.execution.oms import OrderManager
from squeeze_hunter.execution.slicing import build_twap_plan


@pytest.mark.asyncio
async def test_oms_submits_each_slice_in_order() -> None:
    broker = MagicMock()
    submitted = []

    async def fake_submit_buy(ticker, qty, limit_price, ts):
        submitted.append((qty, limit_price))
        return BrokerOrder(
            broker_order_id=f"id-{len(submitted)}",
            ticker=ticker,
            side="buy",
            qty=qty,
            limit_price=limit_price,
            status="filled",
            filled_qty=qty,
            avg_fill_price=limit_price,
        )

    broker.submit_buy = fake_submit_buy
    broker.get_open_orders = AsyncMock(return_value=[])

    open_at = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
    plan = build_twap_plan(
        total_qty=300,
        reference_price=20.0,
        market_open=open_at,
        ticker="GME",
        side="buy",
        n_slices=3,
        window_minutes=10,
        slice_offset_minutes=0,
    )
    oms = OrderManager(broker=broker, clock=lambda: open_at + timedelta(minutes=15))
    result = await oms.execute(plan, max_wall_seconds=0)
    assert len(submitted) == 3
    assert result.filled_qty == 300


@pytest.mark.asyncio
async def test_oms_handles_partial_fill() -> None:
    broker = MagicMock()

    async def fake_submit_buy(ticker, qty, limit_price, ts):
        return BrokerOrder(
            broker_order_id="id-1",
            ticker=ticker,
            side="buy",
            qty=qty,
            limit_price=limit_price,
            status="partial",
            filled_qty=qty // 2,
            avg_fill_price=limit_price,
        )

    broker.submit_buy = fake_submit_buy
    broker.get_open_orders = AsyncMock(return_value=[])

    plan = build_twap_plan(
        total_qty=100,
        reference_price=20.0,
        market_open=datetime(2026, 5, 14, 13, 30, tzinfo=UTC),
        ticker="GME",
        side="buy",
        n_slices=1,
        window_minutes=1,
        slice_offset_minutes=0,
    )
    oms = OrderManager(broker=broker, clock=lambda: datetime(2026, 5, 14, 13, 35, tzinfo=UTC))
    result = await oms.execute(plan, max_wall_seconds=0)
    assert result.filled_qty == 50
    assert result.unfilled_qty == 50
