from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from squeeze_hunter.broker.base import BrokerOrder, Quote
from squeeze_hunter.execution.oms import OrderManager
from squeeze_hunter.execution.slicing import build_twap_plan


def _make_quote(ticker: str, bid: float = 20.0, ask: float = 20.10) -> Quote:
    return Quote(ticker=ticker, bid=bid, ask=ask, last=(bid + ask) / 2, timestamp_ns=0)


@pytest.mark.asyncio
async def test_oms_submits_each_slice_in_order() -> None:
    broker = MagicMock()
    submitted = []

    async def fake_quote(ticker):
        return _make_quote(ticker)

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
    broker.fetch_quote = fake_quote
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

    async def fake_quote(ticker):
        return _make_quote(ticker)

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
    broker.fetch_quote = fake_quote
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


@pytest.mark.asyncio
async def test_oms_repricing_uses_fresh_quote_per_slice() -> None:
    """I4 regression: OMS must compute each slice's limit from a fresh quote,
    not from a stale reference price baked into the plan.
    """
    submitted_limits: list[float] = []
    quotes_per_call = [
        Quote(ticker="GME", bid=20.0, ask=20.05, last=20.0, timestamp_ns=0),
        Quote(ticker="GME", bid=22.0, ask=22.05, last=22.0, timestamp_ns=0),  # price moved up
        Quote(ticker="GME", bid=24.0, ask=24.05, last=24.0, timestamp_ns=0),
    ]

    async def fake_quote(ticker):
        return quotes_per_call.pop(0)

    async def fake_submit_buy(ticker, qty, limit_price, ts):
        submitted_limits.append(limit_price)
        return BrokerOrder(
            broker_order_id=f"id-{len(submitted_limits)}",
            ticker=ticker,
            side="buy",
            qty=qty,
            limit_price=limit_price,
            status="filled",
            filled_qty=qty,
            avg_fill_price=limit_price,
        )

    broker = MagicMock()
    broker.fetch_quote = fake_quote
    broker.submit_buy = fake_submit_buy

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
    await oms.execute(plan, max_wall_seconds=0)

    # Each slice priced off its own quote → limits track the rising market
    assert submitted_limits[0] < submitted_limits[1] < submitted_limits[2]
    # First slice: mid 20.025 + ~5bps ≈ 20.035; last: mid 24.025 + ~30bps ≈ 24.097
    assert submitted_limits[0] < 21.0
    assert submitted_limits[-1] > 23.0


@pytest.mark.asyncio
async def test_oms_escalates_to_marketable_when_fills_lag() -> None:
    """I4: after enough unfilled slices, OMS switches to marketable-limit pricing.

    With a passive limit that doesn't fill (status=cancelled), the next slice
    should use a more aggressive limit price.
    """
    fills_seq = ["cancelled", "cancelled", "cancelled", "cancelled", "filled", "filled"]
    submitted = []

    async def fake_quote(ticker):
        return Quote(ticker=ticker, bid=20.0, ask=20.05, last=20.0, timestamp_ns=0)

    async def fake_submit_buy(ticker, qty, limit_price, ts):
        submitted.append((limit_price, fills_seq[len(submitted)]))
        status = fills_seq[len(submitted) - 1]
        return BrokerOrder(
            broker_order_id=f"id-{len(submitted)}",
            ticker=ticker,
            side="buy",
            qty=qty,
            limit_price=limit_price,
            status=status,
            filled_qty=qty if status == "filled" else 0,
            avg_fill_price=limit_price if status == "filled" else None,
        )

    broker = MagicMock()
    broker.fetch_quote = fake_quote
    broker.submit_buy = fake_submit_buy

    open_at = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
    plan = build_twap_plan(
        total_qty=600,
        reference_price=20.0,
        market_open=open_at,
        ticker="GME",
        side="buy",
        n_slices=6,
        window_minutes=10,
        slice_offset_minutes=0,
    )
    oms = OrderManager(broker=broker, clock=lambda: open_at + timedelta(minutes=15))
    await oms.execute(plan, max_wall_seconds=0)

    # The later slices (after the 80% unfilled threshold) should use higher limits
    early_limits = [lim for lim, _ in submitted[:4]]
    late_limits = [lim for lim, _ in submitted[4:]]
    # All early limits passive (~5-25 bps); late limits marketable (~50+ bps)
    avg_early = sum(early_limits) / len(early_limits)
    avg_late = sum(late_limits) / len(late_limits)
    assert avg_late > avg_early
    # Marketable: at 50bps above mid 20.025 → 20.125
    assert max(submitted[4][0], submitted[5][0]) > 20.10


@pytest.mark.asyncio
async def test_oms_partial_fills_still_escalate() -> None:
    """R9.5 regression: persistent partial fills must still escalate aggression.

    Prior behavior incremented `slices_filled` on ANY filled_qty>0 (including
    partial), so 5 slices each at 30% fill made escalate_aggression think
    fills were full → no marketable escalation → TWAP ended with ~70%
    unfilled at end-of-window.

    Fixed: only fully-filled slices count toward `slices_filled`. After a few
    persistent partials, the marketable-limit threshold trips and later slices
    use a more aggressive limit price.
    """
    submitted_limits: list[float] = []

    async def fake_quote(ticker):
        return Quote(ticker=ticker, bid=20.0, ask=20.05, last=20.0, timestamp_ns=0)

    async def fake_submit_buy(ticker, qty, limit_price, ts):
        submitted_limits.append(limit_price)
        # Every order partially fills 30% — never reaches "filled" status
        return BrokerOrder(
            broker_order_id=f"id-{len(submitted_limits)}",
            ticker=ticker,
            side="buy",
            qty=qty,
            limit_price=limit_price,
            status="partial",
            filled_qty=qty // 3,
            avg_fill_price=limit_price,
        )

    broker = MagicMock()
    broker.fetch_quote = fake_quote
    broker.submit_buy = fake_submit_buy

    open_at = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
    plan = build_twap_plan(
        total_qty=600,
        reference_price=20.0,
        market_open=open_at,
        ticker="GME",
        side="buy",
        n_slices=6,
        window_minutes=10,
        slice_offset_minutes=0,
    )
    oms = OrderManager(broker=broker, clock=lambda: open_at + timedelta(minutes=15))
    await oms.execute(plan, max_wall_seconds=0)

    # The natural aggression ramp goes from ~5 bps to ~30 bps. Escalation to
    # MARKETABLE bumps to 50 bps. If partials are mis-counted as fully filled,
    # escalation never trips and the last slice's limit caps at ~30 bps above
    # mid (=20.025 * 1.0030 ≈ 20.085). With the fix, late slices should hit
    # marketable territory: 20.025 * 1.0050 ≈ 20.125. Use a threshold above
    # the natural-ramp ceiling to detect escalation specifically.
    natural_ramp_ceiling = 20.025 * (1 + 35 / 10_000)  # ~20.095
    assert any(lim > natural_ramp_ceiling for lim in submitted_limits[-2:]), (
        "persistent partial fills did not escalate to marketable; "
        f"limits {submitted_limits} all under natural ramp ceiling {natural_ramp_ceiling:.4f}"
    )


@pytest.mark.asyncio
async def test_oms_avg_fill_price_when_broker_returns_filled_without_price() -> None:
    """Round-11 regression: live IBKRBroker.submit_* returns a BrokerOrder with
    status='filled' but filled_qty=0 / avg_fill_price=None (orderStatus not yet
    populated when placeOrder returns an immediate fill). The OMS counted such
    fills toward cumulative_qty (filled falls back to order.qty) but skipped
    cumulative_value, so result.avg_fill_price was dragged toward 0 — exactly 0.0
    when every slice was priceless — corrupting cost basis / P&L / stop math.
    With no broker price, fall back to the slice limit so qty and value stay
    consistent.
    """

    async def fake_quote(ticker):
        return _make_quote(ticker)  # bid 20.0 / ask 20.10

    async def fake_submit_buy(ticker, qty, limit_price, ts):
        return BrokerOrder(
            broker_order_id="id-1",
            ticker=ticker,
            side="buy",
            qty=qty,
            limit_price=limit_price,
            status="filled",
            filled_qty=0,  # IBKR-shaped: immediate fill, orderStatus not yet synced
            avg_fill_price=None,
        )

    broker = MagicMock()
    broker.fetch_quote = fake_quote
    broker.submit_buy = fake_submit_buy

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
    assert result.filled_qty == 300
    # avg_fill_price must be a real near-market price (~20), NEVER 0.0.
    assert result.avg_fill_price > 0.0
    assert 19.0 < result.avg_fill_price < 21.5


@pytest.mark.asyncio
async def test_oms_marketable_for_sell_uses_negative_aggression() -> None:
    """For sell side, aggression bps means below mid (lower price = more
    aggressive sell)."""
    submitted = []

    async def fake_quote(ticker):
        return Quote(ticker=ticker, bid=100.0, ask=100.10, last=100.0, timestamp_ns=0)

    async def fake_submit_sell(ticker, qty, limit_price, ts):
        submitted.append(limit_price)
        return BrokerOrder(
            broker_order_id=f"id-{len(submitted)}",
            ticker=ticker,
            side="sell",
            qty=qty,
            limit_price=limit_price,
            status="filled",
            filled_qty=qty,
            avg_fill_price=limit_price,
        )

    broker = MagicMock()
    broker.fetch_quote = fake_quote
    broker.submit_sell = fake_submit_sell

    open_at = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
    plan = build_twap_plan(
        total_qty=200,
        reference_price=100.0,
        market_open=open_at,
        ticker="GME",
        side="sell",
        n_slices=2,
        window_minutes=10,
        slice_offset_minutes=0,
    )
    oms = OrderManager(broker=broker, clock=lambda: open_at + timedelta(minutes=15))
    await oms.execute(plan, max_wall_seconds=0)

    # All sell limits should be at-or-below the mid (100.05). First slice
    # passive (close to mid); later slices more aggressive (further below).
    assert all(lim <= 100.10 for lim in submitted)
    # First slice passive: mid - ~5bps = 100.0; last: mid - more
    assert submitted[0] >= submitted[-1]
