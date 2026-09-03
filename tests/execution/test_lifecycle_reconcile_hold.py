"""Round-12 regressions for the lifecycle daemon.

1. A pending exit that filled between ticks must be reconciled even when the
   next tick's stop evaluation says "hold" — otherwise the daemon keeps a
   phantom position forever (every IBKR submit returns "pending" on the
   submitting tick, so this is the normal live path, not an edge case).
2. The signal-decay "halve" must fire once per position. evaluate_stops is
   stateless, so without a flag the live daemon halved again every 60 s
   while decay stayed in [0.50, 0.75).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from squeeze_hunter.broker.base import BrokerOrder, Quote
from squeeze_hunter.execution.lifecycle import LifecycleState, manage_positions

_NOW = datetime(2026, 5, 14, 14, 0, tzinfo=UTC)


def _broker(price: float, sell_status: str = "filled", sell_qty: int = 50) -> MagicMock:
    broker = MagicMock()
    broker.fetch_quote = AsyncMock(
        return_value=Quote(ticker="GME", bid=price, ask=price + 0.05, last=price, timestamp_ns=0)
    )
    broker.submit_sell = AsyncMock(
        return_value=BrokerOrder(
            broker_order_id="sell-1",
            ticker="GME",
            side="sell",
            qty=sell_qty,
            limit_price=price * 0.995,
            status=sell_status,
            filled_qty=sell_qty if sell_status == "filled" else 0,
            avg_fill_price=price if sell_status == "filled" else None,
        )
    )
    broker.cancel_order = AsyncMock(return_value=True)
    broker.get_position_qty = AsyncMock(return_value=100)
    return broker


def _state(**extra: Any) -> LifecycleState:
    meta: dict[str, Any] = {
        "qty": 100,
        "entry_price": 100.0,
        "peak_price": 100.0,
        "entry_score": 10.0,
        "current_score": 10.0,
        "bars_held": 2,
        "setup_type": "CAR",
    }
    meta.update(extra)
    return LifecycleState(positions={"GME": meta})


@pytest.mark.asyncio
async def test_pending_exit_that_filled_is_reconciled_even_when_stop_says_hold() -> None:
    broker = _broker(price=100.0)
    broker.get_position_qty = AsyncMock(return_value=0)  # broker is flat: prior exit filled
    state = _state(pending_exits=["pending-1"], pending_action="exit")

    out = await manage_positions(state=state, broker=broker, now=_NOW)

    assert "GME" not in out.positions, "phantom position retained after the exit filled"
    assert out.exits
    assert out.exits[-1]["reason"] == "reconciled_filled_exit"
    broker.submit_sell.assert_not_called()


@pytest.mark.asyncio
async def test_signal_decay_halve_fires_only_once() -> None:
    broker = _broker(price=100.0, sell_status="filled", sell_qty=50)
    state = _state(current_score=4.0)  # decay 0.6 → halve band

    await manage_positions(state=state, broker=broker, now=_NOW)
    meta = state.positions["GME"]
    assert meta["qty"] == 50
    assert meta.get("halved") is True
    broker.submit_sell.assert_awaited_once()

    # Next tick, decay unchanged: must NOT halve again.
    await manage_positions(state=state, broker=broker, now=_NOW)
    assert state.positions["GME"]["qty"] == 50
    broker.submit_sell.assert_awaited_once()


@pytest.mark.asyncio
async def test_already_halved_position_is_held_in_the_halve_band() -> None:
    broker = _broker(price=100.0)
    state = _state(current_score=4.0, halved=True, qty=50)

    await manage_positions(state=state, broker=broker, now=_NOW)

    broker.submit_sell.assert_not_called()
    assert state.positions["GME"]["qty"] == 50


@pytest.mark.asyncio
async def test_pending_halve_that_filled_is_reconciled_without_resubmitting() -> None:
    """A halve went pending last tick and filled in between: the broker now holds
    50 of the local 100. Reconcile must adopt 50, mark the position halved and
    NOT submit a second halve."""
    broker = _broker(price=100.0)
    broker.get_position_qty = AsyncMock(return_value=50)
    state = _state(current_score=4.0, pending_exits=["halve-1"], pending_action="halve")

    await manage_positions(state=state, broker=broker, now=_NOW)

    meta = state.positions["GME"]
    assert meta["qty"] == 50
    assert meta.get("halved") is True
    assert meta.get("pending_exits", []) == []
    broker.submit_sell.assert_not_called()
