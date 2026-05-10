from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from squeeze_hunter.broker.base import BrokerOrder, Quote
from squeeze_hunter.execution.lifecycle import LifecycleState, manage_positions


@pytest.mark.asyncio
async def test_manage_exits_on_hard_stop() -> None:
    broker = MagicMock()
    broker.fetch_quote = AsyncMock(
        return_value=Quote(ticker="GME", bid=85.0, ask=85.05, last=85.0, timestamp_ns=0)
    )
    broker.submit_sell = AsyncMock(
        return_value=BrokerOrder(
            broker_order_id="x1",
            ticker="GME",
            side="sell",
            qty=100,
            limit_price=None,
            status="filled",
            filled_qty=100,
            avg_fill_price=85.0,
        )
    )
    state = LifecycleState(
        positions={
            "GME": {
                "qty": 100,
                "entry_price": 100.0,
                "peak_price": 110.0,
                "entry_score": 10.0,
                "current_score": 9.0,
                "bars_held": 2,
                "setup_type": "CAR",
            }
        }
    )
    out = await manage_positions(
        state=state, broker=broker, now=datetime(2026, 5, 14, 14, 0, tzinfo=UTC)
    )
    assert "GME" not in out.positions
    assert any(e["reason"] == "hard_stop" for e in out.exits)


@pytest.mark.asyncio
async def test_manage_updates_peak() -> None:
    broker = MagicMock()
    broker.fetch_quote = AsyncMock(
        return_value=Quote(ticker="GME", bid=120.0, ask=120.05, last=120.0, timestamp_ns=0)
    )
    state = LifecycleState(
        positions={
            "GME": {
                "qty": 100,
                "entry_price": 100.0,
                "peak_price": 110.0,
                "entry_score": 10.0,
                "current_score": 10.0,
                "bars_held": 2,
                "setup_type": "CAR",
            }
        }
    )
    out = await manage_positions(
        state=state, broker=broker, now=datetime(2026, 5, 14, 14, 0, tzinfo=UTC)
    )
    assert out.positions["GME"]["peak_price"] == 120.0
