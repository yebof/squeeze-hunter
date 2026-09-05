"""Round-13 runtime regressions: live-port guard, gauges, quote freshness."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from squeeze_hunter.broker.base import Quote
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.monitor.http import MonitorServer
from squeeze_hunter.runtime import RuntimeContext
from tests.runtime.test_session_clamp import _runtime

_IN_SESSION = datetime(2026, 5, 14, 14, 0, tzinfo=UTC)  # Thu 10:00 ET


@pytest.mark.asyncio
async def test_live_mode_refuses_the_paper_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IBKR_PORT unset defaulted to 7497, so `live --confirm-real-money` and
    `emergency-flatten --mode live` quietly talked to the paper account."""
    rc = RuntimeContext(
        cache=ParquetCache(root=tmp_path), settings=Settings(), tickers=["GME"], mode="live"
    )
    monkeypatch.delenv("IBKR_PORT", raising=False)
    with pytest.raises(ValueError, match="IBKR_PORT"):
        await rc.setup()
    monkeypatch.setenv("IBKR_PORT", "7497")
    with pytest.raises(ValueError, match="7497"):
        await rc.setup()


@pytest.mark.asyncio
async def test_gauges_update_even_when_equity_is_unavailable(tmp_path: Path) -> None:
    rc = await _runtime(tmp_path)
    assert rc.broker is not None
    assert rc.metrics_registry is not None
    rc.broker.get_equity_usd = AsyncMock(return_value=None)  # type: ignore[method-assign]
    await rc.tick(now=_IN_SESSION)
    assert "sh_broker_connected 1.0" in rc.metrics_registry.render()


@pytest.mark.asyncio
async def test_quote_freshness_needs_a_fresh_quote_when_positions_exist(tmp_path: Path) -> None:
    rc = await _runtime(tmp_path)
    assert rc.broker is not None
    rc.lifecycle_state.positions["GME"] = {
        "qty": 100,
        "entry_price": 100.0,
        "peak_price": 100.0,
        "entry_score": 10.0,
        "current_score": 10.0,
        "bars_held": 0,
        "setup_type": "CAR",
    }
    rc.broker.fetch_quote = AsyncMock(  # type: ignore[method-assign]
        return_value=Quote(ticker="GME", bid=0.0, ask=0.0, last=0.0, timestamp_ns=0)
    )
    # Last fresh quote before today's open; the session clamp lifts it to 09:30.
    rc.telemetry.record_data_freshness("ibkr_quotes", datetime(2026, 5, 14, 13, 0, tzinfo=UTC))
    await rc.tick(now=_IN_SESSION)
    # Frozen quotes must NOT refresh the stamp; it stays at the session open.
    assert rc.telemetry.data_freshness["ibkr_quotes"] == datetime(2026, 5, 14, 13, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_quote_freshness_refreshes_on_health_when_flat(tmp_path: Path) -> None:
    rc = await _runtime(tmp_path)
    await rc.tick(now=_IN_SESSION)
    assert rc.telemetry.data_freshness["ibkr_quotes"] == _IN_SESSION


def test_monitor_server_stop_before_start_returns_promptly() -> None:
    server = MonitorServer(MagicMock(), host="127.0.0.1", port=0)
    t = threading.Thread(target=server.stop, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()
