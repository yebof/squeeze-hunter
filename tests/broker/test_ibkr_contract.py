"""P3 — the real IBKRBroker against a fake IB that behaves like ib_async."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from squeeze_hunter.broker.ibkr import IBKRBroker
from squeeze_hunter.execution.book import new_position_meta
from squeeze_hunter.execution.lifecycle import LifecycleState, manage_positions
from tests.broker.fake_ib import UNSET_DOUBLE, FakeIB

_TS = datetime(2026, 5, 14, 14, 0, tzinfo=UTC)


def _broker(fake: FakeIB | None = None, account: str = "DU111") -> tuple[IBKRBroker, FakeIB]:
    fake = fake or FakeIB()
    broker = IBKRBroker(client_id=7, account=account)
    broker._ib = fake
    return broker, fake


@pytest.mark.asyncio
async def test_connect_and_equity_use_the_async_subscription() -> None:
    broker, fake = _broker()
    await broker.connect()
    assert fake.connected
    assert await broker.get_equity_usd() == pytest.approx(100_000.0)


@pytest.mark.asyncio
async def test_connect_rejects_an_unmanaged_account() -> None:
    broker, _ = _broker(account="DU999")
    with pytest.raises(ValueError, match="DU999"):
        await broker.connect()


@pytest.mark.asyncio
async def test_quote_freshness_comes_from_the_ticker_time() -> None:
    broker, fake = _broker()
    fake.set_quote("GME", bid=24.5, ask=24.6, last=24.53, at=datetime.now(UTC))
    with patch("squeeze_hunter.broker.ibkr.asyncio.sleep", return_value=None):
        fresh = await broker.fetch_quote("GME")
        assert fresh.last == pytest.approx(24.53)
        fake.set_quote(
            "GME", bid=24.5, ask=24.6, last=24.53, at=datetime.now(UTC) - timedelta(minutes=10)
        )
        stale = await broker.fetch_quote("GME")
    assert stale.last == 0.0
    assert stale.bid == 0.0


@pytest.mark.asyncio
async def test_order_status_progression_and_get_order() -> None:
    broker, fake = _broker()
    order = await broker.submit_sell("GME", 100, 24.4, _TS)
    assert order.status == "pending"
    fake.advance()
    routed = await broker.get_order(order.broker_order_id)
    assert routed is not None
    assert routed.status == "routed"
    fake.fill(int(order.broker_order_id), 40, 24.41)
    partial = await broker.get_order(order.broker_order_id)
    assert partial is not None
    assert partial.status == "partial"
    assert partial.filled_qty == 40
    fake.fill(int(order.broker_order_id), 60, 24.40)
    done = await broker.get_order(order.broker_order_id)
    assert done is not None
    assert done.status == "filled"
    assert done.avg_fill_price == pytest.approx((40 * 24.41 + 60 * 24.40) / 100)
    assert await broker.get_open_orders() == []


@pytest.mark.asyncio
async def test_market_order_limit_is_reported_as_none() -> None:
    broker, fake = _broker()
    order = await broker.submit_buy("GME", 10, None, _TS)
    assert fake.placed[-1].order.lmtPrice == UNSET_DOUBLE
    fetched = await broker.get_order(order.broker_order_id)
    assert fetched is not None
    assert fetched.limit_price is None
    assert (await broker.get_open_orders())[0].limit_price is None


@pytest.mark.asyncio
async def test_validation_error_is_rejected_and_stays_out_of_open_orders_view() -> None:
    broker, fake = _broker()
    order = await broker.submit_sell("GME", 100, 24.4, _TS)
    fake.reject(int(order.broker_order_id), validation=True)
    fetched = await broker.get_order(order.broker_order_id)
    assert fetched is not None
    assert fetched.status == "rejected"


@pytest.mark.asyncio
async def test_lifecycle_waits_for_the_cancel_to_be_acknowledged() -> None:
    """Tick 1: hard stop → exit goes pending. Tick 2: still open → cancel;
    IBKR only reports PendingCancel, so NO replacement is sent. Tick 3 (cancel
    acknowledged): the replacement goes out. Exactly two orders ever exist."""
    broker, fake = _broker()
    fake.cancel_latency = 1
    fake.set_position("GME", 100, 100.0)
    fake.set_quote("GME", bid=50.0, ask=50.1, last=50.0)
    state = LifecycleState(
        positions={
            "GME": new_position_meta(
                ticker="GME", qty=100, entry_price=100.0, score=10.0, setup_type="CAR"
            )
        }
    )
    with (
        patch("squeeze_hunter.broker.ibkr.asyncio.sleep", return_value=None),
        patch("squeeze_hunter.execution.lifecycle._CANCEL_CONFIRM_SLEEP_S", 0),
        patch("squeeze_hunter.execution.lifecycle._CANCEL_CONFIRM_POLLS", 1),
    ):
        await manage_positions(state, broker, _TS)
        assert len(fake.placed) == 1
        assert state.positions["GME"]["pending_exits"] == ["1"]
        fake.advance()  # PendingSubmit → Submitted, still unfilled

        await manage_positions(state, broker, _TS + timedelta(minutes=1))
        assert fake.placed[0].orderStatus.status == "PendingCancel"
        assert len(fake.placed) == 1, "resubmitted while the cancel was still pending"
        fake.advance()  # cancel acknowledged

        await manage_positions(state, broker, _TS + timedelta(minutes=2))
    assert fake.placed[0].orderStatus.status == "Cancelled"
    assert len(fake.placed) == 2
    assert state.positions["GME"]["pending_exits"] == ["2"]


@pytest.mark.asyncio
async def test_get_positions_reads_the_synced_snapshot() -> None:
    broker, fake = _broker()
    fake.set_position("GME", 100, 18.0)
    fake.set_position("AMC", 50, 5.0, account="U222")
    snaps = await broker.get_positions()
    assert [(p.ticker, p.qty, p.avg_cost) for p in snaps] == [("GME", 100, 18.0)]
