import asyncio
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


@pytest.mark.asyncio
async def test_lifecycle_concurrent_calls_dont_double_sell() -> None:
    """I5 regression: two overlapping manage_positions calls on the same
    state must not both submit a sell for the same position. The second
    caller should observe the in-flight ticker and skip.

    Setup: a position with a hard stop pending, a slow fetch_quote that
    yields. Two concurrent ticks. Only one sell should be submitted.
    """
    sells: list[tuple[str, int]] = []

    async def slow_quote(ticker):
        await asyncio.sleep(0.05)  # yields, allowing the other task to interleave
        return Quote(ticker=ticker, bid=80.0, ask=80.05, last=80.0, timestamp_ns=0)

    async def record_sell(ticker, qty, limit_price, ts):
        sells.append((ticker, qty))
        return BrokerOrder(
            broker_order_id=f"x-{len(sells)}",
            ticker=ticker,
            side="sell",
            qty=qty,
            limit_price=limit_price,
            status="filled",
            filled_qty=qty,
            avg_fill_price=80.0,
        )

    broker = MagicMock()
    broker.fetch_quote = slow_quote
    broker.submit_sell = record_sell

    state = LifecycleState(
        positions={
            "GME": {
                "qty": 100,
                "entry_price": 100.0,
                "peak_price": 100.0,
                "entry_score": 10.0,
                "current_score": 10.0,
                "bars_held": 2,
                "setup_type": "CAR",
            }
        }
    )
    # Two concurrent calls
    await asyncio.gather(
        manage_positions(state, broker, datetime(2026, 5, 14, 14, 0, tzinfo=UTC)),
        manage_positions(state, broker, datetime(2026, 5, 14, 14, 0, tzinfo=UTC)),
    )
    # Only ONE sell — the second call must observe in-flight and skip
    assert len(sells) == 1


@pytest.mark.asyncio
async def test_lifecycle_concurrent_different_tickers_processed_independently() -> None:
    """The lock is per-ticker. Two tickers can be processed in parallel without
    blocking each other.
    """
    quotes_fetched = []

    async def slow_quote(ticker):
        quotes_fetched.append(ticker)
        await asyncio.sleep(0.05)
        return Quote(ticker=ticker, bid=100.0, ask=100.05, last=100.0, timestamp_ns=0)

    broker = MagicMock()
    broker.fetch_quote = slow_quote
    broker.submit_sell = AsyncMock()

    state_a = LifecycleState(
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
    state_b = LifecycleState(
        positions={
            "AAPL": {
                "qty": 50,
                "entry_price": 200.0,
                "peak_price": 200.0,
                "entry_score": 10.0,
                "current_score": 10.0,
                "bars_held": 1,
                "setup_type": "CAR",
            }
        }
    )
    # Different state objects → independent locks → both fetch
    start = asyncio.get_event_loop().time()
    await asyncio.gather(
        manage_positions(state_a, broker, datetime(2026, 5, 14, 14, 0, tzinfo=UTC)),
        manage_positions(state_b, broker, datetime(2026, 5, 14, 14, 0, tzinfo=UTC)),
    )
    duration = asyncio.get_event_loop().time() - start
    # If they ran sequentially: 2 x 0.05 = 0.1s. Concurrently: ~0.05s.
    assert duration < 0.09  # generous bound for CI noise
    assert len(quotes_fetched) == 2


@pytest.mark.asyncio
async def test_lifecycle_in_flight_set_is_cleared_after_processing() -> None:
    """After manage_positions completes (success or failure), in_flight
    must be empty so subsequent ticks can process the same tickers.
    """
    broker = MagicMock()
    broker.fetch_quote = AsyncMock(
        return_value=Quote(
            ticker="GME",
            bid=100.0,
            ask=100.05,
            last=100.0,
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
    await manage_positions(state, broker, datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    assert state.in_flight == set()


@pytest.mark.asyncio
async def test_lifecycle_in_flight_cleared_on_exception() -> None:
    """If processing raises (e.g., AttributeError per I3), in_flight must still
    be cleared so the next tick can retry.
    """
    broker = MagicMock()
    broker.fetch_quote = AsyncMock(side_effect=AttributeError("boom"))

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
    with pytest.raises(AttributeError):
        await manage_positions(state, broker, datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    # Even after the exception, in_flight is empty
    assert state.in_flight == set()


def test_lifecycle_exits_list_trims_after_max() -> None:
    """R5.I1 regression: state.exits is bounded so it doesn't grow unbounded
    over long runs. With 60s ticks and ~390 exits/day possible in the worst
    case, the list could otherwise grow into tens of thousands of entries
    over months of paper trading.
    """
    state = LifecycleState(exits_max_entries=10)
    for i in range(25):
        state.record_exit(
            {
                "ts": datetime(2026, 5, 14, 14, 0, tzinfo=UTC),
                "ticker": f"T{i}",
                "qty": 1,
                "reason": "test",
            }
        )
    assert len(state.exits) == 10
    # Most recent 10 entries kept (oldest dropped)
    tickers = [e["ticker"] for e in state.exits]
    assert tickers == [f"T{i}" for i in range(15, 25)]


def test_lifecycle_exits_default_max_is_generous() -> None:
    """Default cap is high (1000) so it never bites in normal operation;
    only catastrophic runaway scenarios would hit it."""
    state = LifecycleState()
    assert state.exits_max_entries >= 1000
