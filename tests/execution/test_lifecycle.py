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


@pytest.mark.asyncio
async def test_lifecycle_skips_position_when_quote_is_zero() -> None:
    """I2 regression: zero-price quote (halt / stale data) must NOT trigger a hard stop.

    Before the fix: q.last=0 + q.bid=0 + q.ask=0 → price=0 → pnl_pct=-1 → hard
    stop triggers → submit_sell market order at the next print (potentially
    catastrophic during a halt).

    After the fix: lifecycle skips the position this tick; no order submitted.
    """
    broker = MagicMock()
    broker.fetch_quote = AsyncMock(
        return_value=Quote(
            ticker="GME",
            bid=0.0,
            ask=0.0,
            last=0.0,
            timestamp_ns=0,
        )
    )
    broker.submit_sell = AsyncMock()
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
    out = await manage_positions(state, broker, datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    # Position still held — no stop fired
    assert "GME" in out.positions
    # No sell submitted
    assert not broker.submit_sell.called


@pytest.mark.asyncio
async def test_lifecycle_attribute_error_propagates() -> None:
    """I3 regression: AttributeError (programming bug, e.g. broker missing
    a method) must NOT be silently swallowed. The lifecycle should let it
    propagate so the supervising layer (tick_safe) logs it loudly.
    """
    broker = MagicMock()
    broker.fetch_quote = AsyncMock(
        side_effect=AttributeError("'SomeBroker' object has no attribute 'fetch_quote'")
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
    with pytest.raises(AttributeError):
        await manage_positions(state, broker, datetime(2026, 5, 14, 14, 0, tzinfo=UTC))


@pytest.mark.asyncio
async def test_lifecycle_not_implemented_propagates() -> None:
    """Same for NotImplementedError — also a programming/contract issue."""
    broker = MagicMock()
    broker.fetch_quote = AsyncMock(side_effect=NotImplementedError("stub"))
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
    with pytest.raises(NotImplementedError):
        await manage_positions(state, broker, datetime(2026, 5, 14, 14, 0, tzinfo=UTC))


@pytest.mark.asyncio
async def test_lifecycle_connection_error_caught_and_continues() -> None:
    """I3 (positive case): ConnectionError IS transient — log and continue,
    don't propagate. Other tickers in the same tick should still be processed.
    """
    broker = MagicMock()

    quote_calls = []

    async def maybe_fail(ticker):
        quote_calls.append(ticker)
        if ticker == "BAD":
            raise ConnectionError("transient")
        return Quote(ticker=ticker, bid=100.0, ask=100.05, last=100.0, timestamp_ns=0)

    broker.fetch_quote = maybe_fail
    broker.submit_sell = AsyncMock()
    state = LifecycleState(
        positions={
            "BAD": {
                "qty": 50,
                "entry_price": 100.0,
                "peak_price": 100.0,
                "entry_score": 10.0,
                "current_score": 10.0,
                "bars_held": 1,
                "setup_type": "CAR",
            },
            "GOOD": {
                "qty": 100,
                "entry_price": 100.0,
                "peak_price": 100.0,
                "entry_score": 10.0,
                "current_score": 10.0,
                "bars_held": 1,
                "setup_type": "CAR",
            },
        }
    )
    # Should NOT raise — transient error swallowed
    out = await manage_positions(state, broker, datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    # Both positions still held; both fetches attempted
    assert "BAD" in out.positions
    assert "GOOD" in out.positions
    assert set(quote_calls) >= {"BAD", "GOOD"}


@pytest.mark.asyncio
async def test_lifecycle_partial_zero_price_uses_nonzero_field() -> None:
    """If only `last` is 0 but `bid` is positive, use bid (or ask).
    This is a common condition right after open before last-trade prints.
    """
    broker = MagicMock()
    broker.fetch_quote = AsyncMock(
        return_value=Quote(
            ticker="GME",
            bid=100.0,
            ask=100.10,
            last=0.0,
            timestamp_ns=0,
        )
    )
    broker.submit_sell = AsyncMock()
    state = LifecycleState(
        positions={
            "GME": {
                "qty": 100,
                "entry_price": 100.0,
                "peak_price": 100.0,
                "entry_score": 10.0,
                "current_score": 10.0,
                "bars_held": 1,
                "setup_type": "CAR",
            }
        }
    )
    out = await manage_positions(state, broker, datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    # No stop fires — bid 100 = entry 100, no adverse move
    assert "GME" in out.positions
    assert not broker.submit_sell.called
