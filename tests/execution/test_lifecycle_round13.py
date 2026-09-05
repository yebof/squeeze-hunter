"""Round-13 regressions for the lifecycle daemon (live money path)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.broker.base import BrokerOrder, Quote
from squeeze_hunter.execution.lifecycle import manage_positions
from tests.execution.test_lifecycle_reconcile_hold import _NOW, _broker, _state


@pytest.mark.asyncio
async def test_exit_limit_is_rounded_down_to_the_penny() -> None:
    """24.53 * 0.995 = 24.40735: IBKR rejects sub-penny limits (error 110) and
    the daemon would resubmit the same bad price forever."""
    broker = _broker(price=24.53)
    state = _state(entry_price=100.0)  # -75% → hard stop
    await manage_positions(state=state, broker=broker, now=_NOW)
    broker.submit_sell.assert_awaited_once()
    assert broker.submit_sell.await_args.kwargs["limit_price"] == pytest.approx(24.40)


@pytest.mark.asyncio
async def test_sub_dollar_exit_limit_keeps_four_decimals() -> None:
    broker = _broker(price=0.90)
    state = _state(entry_price=100.0)
    await manage_positions(state=state, broker=broker, now=_NOW)
    assert broker.submit_sell.await_args.kwargs["limit_price"] == pytest.approx(0.8955)


@pytest.mark.asyncio
async def test_reconcile_runs_before_the_quote_guard() -> None:
    """A filled exit must be reconciled even while quotes are NaN (halt)."""
    broker = _broker(price=float("nan"))
    broker.get_position_qty = AsyncMock(return_value=0)
    state = _state(pending_exits=["p-1"], pending_action="exit")
    await manage_positions(state=state, broker=broker, now=_NOW)
    assert "GME" not in state.positions
    assert state.exits[-1]["reason"] == "reconciled_filled_exit"


@pytest.mark.asyncio
async def test_partially_filled_halve_cancels_the_working_remainder() -> None:
    """Halve of 50 went pending; 20 filled; now the stops say exit. The
    remaining 30-share halve order must be cancelled before the 80-share exit
    goes out, or a fill on both would leave the account short."""
    broker = _broker(price=50.0)  # entry 100 → hard stop → exit
    broker.get_position_qty = AsyncMock(return_value=80)
    state = _state(pending_exits=["h-1"], pending_action="halve", pending_qty=50)

    await manage_positions(state=state, broker=broker, now=_NOW)

    broker.cancel_order.assert_awaited_once_with("h-1")
    broker.submit_sell.assert_awaited_once()
    assert broker.submit_sell.await_args.kwargs["qty"] == 80


@pytest.mark.asyncio
async def test_one_share_position_marks_halved_instead_of_looping() -> None:
    broker = _broker(price=100.0)
    state = _state(qty=1, current_score=4.0)  # halve band, but 1 // 2 == 0
    await manage_positions(state=state, broker=broker, now=_NOW)
    broker.submit_sell.assert_not_called()
    assert state.positions["GME"].get("halved") is True
    await manage_positions(state=state, broker=broker, now=_NOW)
    broker.submit_sell.assert_not_called()


@pytest.mark.asyncio
async def test_rejected_exit_is_not_tracked_as_pending() -> None:
    broker = _broker(price=50.0, sell_status="rejected")
    state = _state()
    await manage_positions(state=state, broker=broker, now=_NOW)
    meta = state.positions["GME"]
    assert meta.get("pending_exits", []) == []
    assert "pending_action" not in meta


class _PollingBroker:
    """Minimal broker whose cancel is asynchronous: the order stays in
    get_open_orders() for `open_polls` polls after cancel_order()."""

    def __init__(self, open_polls: int) -> None:
        self.open_polls = open_polls
        self.polls = 0
        self.cancelled: list[str] = []
        self.submitted: list[int] = []

    async def fetch_quote(self, ticker: str) -> Quote:
        return Quote(ticker=ticker, bid=50.0, ask=50.05, last=50.0, timestamp_ns=0)

    async def get_position_qty(self, ticker: str) -> int:
        return 100

    async def cancel_order(self, broker_order_id: str) -> bool:
        self.cancelled.append(broker_order_id)
        return True

    async def get_open_orders(self) -> list[BrokerOrder]:
        self.polls += 1
        if self.polls <= self.open_polls:
            return [
                BrokerOrder(
                    broker_order_id="s-1",
                    ticker="GME",
                    side="sell",
                    qty=100,
                    limit_price=49.0,
                    status="pending",
                    filled_qty=0,
                    avg_fill_price=None,
                )
            ]
        return []

    async def submit_sell(
        self, ticker: str, qty: int, limit_price: float, ts: datetime
    ) -> BrokerOrder:
        self.submitted.append(qty)
        return BrokerOrder(
            broker_order_id="s-2",
            ticker=ticker,
            side="sell",
            qty=qty,
            limit_price=limit_price,
            status="filled",
            filled_qty=qty,
            avg_fill_price=limit_price,
        )


@pytest.mark.asyncio
async def test_resubmit_waits_until_the_cancel_is_terminal() -> None:
    broker: Any = _PollingBroker(open_polls=2)
    state = _state(pending_exits=["s-1"], pending_action="exit")  # entry 100, price 50 → exit
    await manage_positions(state=state, broker=broker, now=_NOW)
    assert broker.cancelled == ["s-1"]
    assert broker.submitted == [100]
    assert broker.polls >= 3


@pytest.mark.asyncio
async def test_no_resubmit_while_the_cancel_is_still_pending() -> None:
    broker: Any = _PollingBroker(open_polls=10**6)
    state = _state(pending_exits=["s-1"], pending_action="exit")
    await manage_positions(state=state, broker=broker, now=_NOW)
    assert broker.cancelled == ["s-1"]
    assert broker.submitted == []
    assert state.positions["GME"]["pending_exits"] == ["s-1"]
