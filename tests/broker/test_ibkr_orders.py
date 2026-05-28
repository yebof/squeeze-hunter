from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

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
async def test_fetch_quote_coalesces_nan_to_zero_when_no_data() -> None:
    """Round-11 regression: ib-async's Ticker initializes bid/ask/last/close to
    float('nan'), NOT None. Because nan is truthy, `float(ticker_data.bid or 0.0)`
    returned nan whenever market data had not arrived (halt / after-hours /
    unsubscribed / slow snapshot) — the intended coalescing to 0.0 never happened.
    NaN then slips past the lifecycle's `price <= 0.0` stop guard (nan <= 0.0 is
    False), silently disabling the hard/trailing stops. fetch_quote must coalesce
    non-finite values to 0.0 so the documented zero-price guard fires.
    """
    broker = IBKRBroker(client_id=99)
    fake_ib = MagicMock()
    fake_ib.qualifyContractsAsync = AsyncMock()
    nan = float("nan")
    ticker_data = MagicMock()
    ticker_data.bid = nan
    ticker_data.ask = nan
    ticker_data.last = nan
    ticker_data.close = nan
    fake_ib.reqMktData = MagicMock(return_value=ticker_data)
    broker._ib = fake_ib

    # Patch sleep so the (now correctly-waiting) snapshot loop doesn't take 10s.
    with patch("squeeze_hunter.broker.ibkr.asyncio.sleep", new=AsyncMock()):
        q = await broker.fetch_quote("GME")

    assert q.bid == 0.0
    assert q.ask == 0.0
    assert q.last == 0.0
    # The lifecycle zero-price guard (`price <= 0.0`) must now fire.
    price = q.last or q.bid or q.ask
    assert price <= 0.0


@pytest.mark.asyncio
async def test_fetch_quote_falls_back_to_close_when_last_is_nan() -> None:
    """Round-11: when `last` and bid/ask are nan (no live trade yet) but the
    prior `close` is valid, fetch_quote should report the close — not 0.0 and
    not nan."""
    broker = IBKRBroker(client_id=99)
    fake_ib = MagicMock()
    fake_ib.qualifyContractsAsync = AsyncMock()
    nan = float("nan")
    ticker_data = MagicMock()
    ticker_data.bid = nan
    ticker_data.ask = nan
    ticker_data.last = nan
    ticker_data.close = 21.5
    fake_ib.reqMktData = MagicMock(return_value=ticker_data)
    broker._ib = fake_ib

    with patch("squeeze_hunter.broker.ibkr.asyncio.sleep", new=AsyncMock()):
        q = await broker.fetch_quote("GME")

    assert q.last == 21.5


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


def _capture_ib_with_qualify(captured: list) -> MagicMock:
    """Helper: build a fake IB whose qualifyContractsAsync records the contract
    and whose placeOrder returns a benign trade (so submit_buy/submit_sell run
    end-to-end)."""
    fake_ib = MagicMock()
    fake_trade = MagicMock()
    fake_trade.order.orderId = 99
    fake_trade.orderStatus.status = "PreSubmitted"
    fake_ib.placeOrder = MagicMock(return_value=fake_trade)

    async def _qualify(contract):
        captured.append(contract)

    fake_ib.qualifyContractsAsync = _qualify
    fake_ib.reqMktData = MagicMock()
    return fake_ib


@pytest.mark.asyncio
async def test_known_nyse_ticker_carries_nyse_primary_exchange() -> None:
    """R9.6 regression: tickers in _PRIMARY_EXCHANGE_DEFAULTS get the explicit
    listing exchange so qualifyContractsAsync uniquely matches the contract
    (avoids picking up a delisted-then-reused shadow contract)."""
    broker = IBKRBroker(client_id=99)
    captured: list = []
    broker._ib = _capture_ib_with_qualify(captured)
    await broker.submit_buy(
        ticker="GME", qty=10, limit_price=5.0, ts=datetime(2026, 5, 14, tzinfo=UTC)
    )
    assert len(captured) == 1
    # GME is in the defaults map as NYSE — must NOT be NASDAQ or empty.
    assert (
        getattr(captured[0], "primaryExchange", "") == "NYSE"
    ), f"GME contract primaryExchange should be NYSE, got {captured[0]!r}"


@pytest.mark.asyncio
async def test_known_nasdaq_ticker_carries_nasdaq_primary_exchange() -> None:
    """R10.1: tickers KNOWN to list on NASDAQ get the explicit hint so
    delisted-then-reused symbols (BBBY, etc.) don't match the wrong contract.
    """
    broker = IBKRBroker(client_id=99)
    captured: list = []
    broker._ib = _capture_ib_with_qualify(captured)
    await broker.submit_sell(
        ticker="BBBY", qty=10, limit_price=5.0, ts=datetime(2026, 5, 14, tzinfo=UTC)
    )
    assert getattr(captured[0], "primaryExchange", "") == "NASDAQ"


@pytest.mark.asyncio
async def test_unknown_ticker_falls_back_to_smart_no_primary_exchange() -> None:
    """R10.1 regression: tickers WITHOUT an explicit defaults entry must NOT
    receive a hardcoded primaryExchange — supplying the wrong one (e.g.,
    NASDAQ for an NYSE-listed F or BAC) makes qualifyContractsAsync return
    zero matches and the order fails to submit. Plain Stock(t,"SMART","USD")
    lets SMART auto-disambiguate the safe majority of unambiguous symbols.
    """
    broker = IBKRBroker(client_id=99)
    captured: list = []
    broker._ib = _capture_ib_with_qualify(captured)
    await broker.submit_buy(
        ticker="F", qty=10, limit_price=12.0, ts=datetime(2026, 5, 14, tzinfo=UTC)
    )
    # Empty / missing primaryExchange is the correct fallback.
    assert getattr(captured[0], "primaryExchange", "") == ""


@pytest.mark.asyncio
async def test_per_instance_override_wins_over_defaults_and_can_force_blank() -> None:
    """R10.1: operator-supplied overrides take precedence. An explicit
    empty-string override forces SMART auto-disambiguation, even for a
    ticker that has a default entry (e.g., when the operator's account
    routes differently)."""
    # Override BBBY (NASDAQ default) to "" → SMART auto-disambiguate.
    # Override AAPL (no default) to "ARCA" → explicit ARCA primary.
    broker = IBKRBroker(client_id=99, primary_exchange_overrides={"BBBY": "", "AAPL": "ARCA"})
    captured: list = []
    broker._ib = _capture_ib_with_qualify(captured)
    await broker.submit_buy(
        ticker="BBBY", qty=10, limit_price=5.0, ts=datetime(2026, 5, 14, tzinfo=UTC)
    )
    await broker.submit_buy(
        ticker="AAPL", qty=10, limit_price=200.0, ts=datetime(2026, 5, 14, tzinfo=UTC)
    )
    assert getattr(captured[0], "primaryExchange", "") == ""
    assert getattr(captured[1], "primaryExchange", "") == "ARCA"


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


@pytest.mark.asyncio
async def test_ibkr_get_position_qty_refreshes_positions_snapshot() -> None:
    """CDX2-P1 regression: ib-async's IB.positions() is a cached snapshot fed
    by TWS push events; right after a fill it can still show the OLD position.
    get_position_qty MUST reqPositionsAsync() to force a fresh sync BEFORE
    reading positions() — otherwise the lifecycle pending-exit reconcile
    (CDX-P1-3) reads a stale qty, concludes "didn't fill", and resubmits the
    full sell → the exact double-sell/short the reconcile was meant to stop.
    """
    broker = IBKRBroker(client_id=99)
    fake_ib = MagicMock()
    order: list[str] = []

    async def fake_req_positions():
        order.append("reqPositionsAsync")

    def fake_positions():
        order.append("positions")
        pos = MagicMock()
        pos.contract.symbol = "GME"
        pos.position = 40
        return [pos]

    fake_ib.reqPositionsAsync = fake_req_positions
    fake_ib.positions = fake_positions
    broker._ib = fake_ib

    qty = await broker.get_position_qty("GME")
    assert qty == 40
    # The refresh MUST happen before the read.
    assert order == [
        "reqPositionsAsync",
        "positions",
    ], f"positions() read without a fresh reqPositionsAsync() sync: {order}"


@pytest.mark.asyncio
async def test_ibkr_get_position_qty_timeout_is_transient() -> None:
    """CDX2-P1: if reqPositionsAsync() hangs (wedged TWS), get_position_qty
    must time out and raise a TRANSIENT error (TimeoutError) rather than block
    the reconcile forever. lifecycle's reconcile catches transient errors and
    conservatively skips the resubmit (no shorting).
    """
    import asyncio

    broker = IBKRBroker(client_id=99)
    fake_ib = MagicMock()

    async def hang():
        await asyncio.sleep(60)

    fake_ib.reqPositionsAsync = hang
    fake_ib.positions = MagicMock(return_value=[])
    broker._ib = fake_ib

    with pytest.raises(TimeoutError):
        await broker.get_position_qty("GME", refresh_timeout_s=0.05)
