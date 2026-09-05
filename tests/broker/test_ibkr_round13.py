"""Round-13 regressions for IBKRBroker against ib_async's real behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from squeeze_hunter.broker.ibkr import IBKRBroker, _translate_status


def _fake_ib(managed: tuple[str, ...] = ("DU111",), values: list | None = None) -> MagicMock:
    ib = MagicMock()
    ib.connectAsync = AsyncMock()
    ib.reqAccountUpdatesAsync = AsyncMock()
    # ib_async's sync variant runs loop.run_until_complete → RuntimeError
    # inside a running loop; the broker must never call it.
    ib.reqAccountUpdates = MagicMock(side_effect=RuntimeError("This event loop is already running"))
    ib.managedAccounts = MagicMock(return_value=list(managed))
    ib.client.serverVersion = MagicMock(return_value=176)
    ib.accountValues = MagicMock(return_value=values or [])
    ib.isConnected = MagicMock(return_value=True)
    ib.qualifyContractsAsync = AsyncMock()
    return ib


@pytest.mark.asyncio
async def test_connect_subscribes_to_account_updates_without_blocking() -> None:
    broker = IBKRBroker(client_id=1, account="DU111")
    broker._ib = _fake_ib()
    await broker.connect()
    broker._ib.reqAccountUpdatesAsync.assert_awaited_once()
    broker._ib.reqAccountUpdates.assert_not_called()


@pytest.mark.asyncio
async def test_connect_refuses_an_account_the_login_does_not_manage() -> None:
    """IBKR_ACCOUNT=DU0000000 copied from .env.example would otherwise make
    get_position_qty() see zero shares and the daemon pop real positions."""
    broker = IBKRBroker(client_id=1, account="DU999")
    broker._ib = _fake_ib(managed=("DU111",))
    with pytest.raises(ValueError, match="DU999"):
        await broker.connect()


@pytest.mark.asyncio
async def test_get_equity_reads_net_liquidation_without_blocking() -> None:
    nav = SimpleNamespace(tag="NetLiquidation", value="12345.6", currency="USD", account="DU111")
    broker = IBKRBroker(client_id=1, account="DU111")
    broker._ib = _fake_ib(values=[nav])
    assert await broker.get_equity_usd() == pytest.approx(12345.6)
    broker._ib.reqAccountUpdates.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_quote_rejects_a_frozen_ticker() -> None:
    """ib_async hands back the SAME Ticker object on every reqMktData; its old
    finite values pass the isfinite check immediately. A ticker whose last
    update is older than the freshness budget must come back as zeros so the
    lifecycle guard skips it."""
    broker = IBKRBroker(client_id=1)
    ib = _fake_ib()
    ticker = SimpleNamespace(
        last=24.53, bid=24.5, ask=24.6, close=24.0, time=datetime.now(UTC) - timedelta(minutes=10)
    )
    ib.reqMktData = MagicMock(return_value=ticker)
    broker._ib = ib

    stale = await broker.fetch_quote("GME")
    assert stale.last == 0.0
    assert stale.bid == 0.0

    ticker.time = datetime.now(UTC)
    fresh = await broker.fetch_quote("GME")
    assert fresh.last == pytest.approx(24.53)


def test_validation_error_status_is_not_pending() -> None:
    assert _translate_status("ValidationError") == "rejected"
