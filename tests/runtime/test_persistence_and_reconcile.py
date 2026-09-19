"""P2 — restart survival and broker reconciliation.

Before this, positions, pending orders and the killswitch lockout lived only
in memory: a restart orphaned real exposure at the broker and erased a
7-day cooldown. Now the runtime persists a snapshot after every job and
reconciles the book against the broker at startup, every tick and at EOD.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.base import PositionSnapshot
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.execution.book import new_position_meta
from squeeze_hunter.monitor.alerts import AlertSender
from squeeze_hunter.runtime import RuntimeContext
from squeeze_hunter.store.state import JsonStateStore
from tests.runtime.test_session_clamp import _seed

_IN_SESSION = datetime(2026, 5, 14, 14, 0, tzinfo=UTC)


def _settings() -> Settings:
    s = Settings()
    s.score.weights = {"f6_bollinger_breakout": 1.0, "f7_volume_spike": 1.0}
    return s


def _cache(tmp_path: Path) -> ParquetCache:
    cache = ParquetCache(root=tmp_path / "parquet")
    _seed(cache)
    return cache


def _rc(tmp_path: Path, broker: SimulatorBroker, store: JsonStateStore) -> RuntimeContext:
    return RuntimeContext(
        cache=_cache(tmp_path),
        settings=_settings(),
        tickers=["GME"],
        mode="sim",
        broker=broker,
        state_store=store,
    )


@pytest.mark.asyncio
async def test_positions_and_killswitch_survive_a_restart(tmp_path: Path) -> None:
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    await broker.submit_buy("GME", 100, 18.0, _IN_SESSION)
    store = JsonStateStore(tmp_path / "state.json")

    rc1 = _rc(tmp_path, broker, store)
    await rc1.setup()
    rc1.lifecycle_state.positions["GME"] = new_position_meta(
        ticker="GME", qty=100, entry_price=18.0, score=9.0, setup_type="CAR"
    )
    rc1.lifecycle_state.positions["GME"]["pending_exits"] = ["sell-1"]
    rc1.lifecycle_state.positions["GME"]["pending_action"] = "exit"
    rc1.kill_switch_active = True
    rc1._kill_reason = "monthly_drawdown"
    rc1._kill_first_tripped_at = _IN_SESSION - timedelta(days=1)
    rc1.telemetry.record_equity(_IN_SESSION - timedelta(days=1), 90_000.0)
    await rc1.eod_close(now=datetime(2026, 5, 14, 20, 30, tzinfo=UTC))  # persists
    assert (tmp_path / "state.json").is_file()

    # "kill -9": a brand-new context over the same broker and store.
    rc2 = _rc(tmp_path, broker, store)
    await rc2.setup()
    meta = rc2.lifecycle_state.positions["GME"]
    assert meta["qty"] == 100
    assert meta["entry_price"] == 18.0
    assert meta["setup_type"] == "CAR"
    assert meta["bars_held"] == 1  # eod_close ran once before the restart
    assert meta["pending_exits"] == ["sell-1"]
    assert rc2.kill_switch_active
    assert rc2._kill_reason == "monthly_drawdown"
    assert rc2._kill_first_tripped_at == _IN_SESSION - timedelta(days=1)
    assert rc2.telemetry.equity_history[0][1] == 90_000.0
    assert "GME" in rc2.telemetry.position_marks


@pytest.mark.asyncio
async def test_startup_adopts_unknown_broker_positions_and_drops_phantoms(tmp_path: Path) -> None:
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    await broker.submit_buy("GME", 40, 20.0, _IN_SESSION)  # at the broker, unknown locally
    store = JsonStateStore(tmp_path / "state.json")
    store.save(
        {
            "version": 1,
            "positions": {
                "AMC": new_position_meta(
                    ticker="AMC", qty=10, entry_price=5.0, score=9.0, setup_type="GME"
                )
            },
        }
    )
    rc = _rc(tmp_path, broker, store)
    rc.alerts = AlertSender(telegram_bot_token="t", telegram_chat_id="c", slack_webhook_url=None)
    rc.alerts._send_telegram = AsyncMock()  # type: ignore[method-assign]
    await rc.setup()

    assert "AMC" not in rc.lifecycle_state.positions  # phantom dropped
    adopted = rc.lifecycle_state.positions["GME"]
    assert adopted["qty"] == 40
    assert adopted["entry_price"] == pytest.approx(broker.positions["GME"].avg_price)
    assert adopted["setup_type"] == "Mixed"
    assert adopted["entry_score"] == 0.0  # signal-decay stops disabled for an adopted lot
    rc.alerts._send_telegram.assert_awaited_once()
    text = rc.alerts._send_telegram.await_args.args[0]
    assert "GME" in text
    assert "AMC" in text


@pytest.mark.asyncio
async def test_tick_reconcile_adopts_broker_quantity(tmp_path: Path) -> None:
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    await broker.submit_buy("GME", 100, 18.0, _IN_SESSION)
    rc = _rc(tmp_path, broker, JsonStateStore(tmp_path / "state.json"))
    await rc.setup()
    rc.lifecycle_state.positions["GME"]["qty"] = 100
    # Something sold 40 shares behind our back (manual trade in TWS).
    await broker.submit_sell("GME", 40, 18.0, _IN_SESSION)
    await rc.tick(now=_IN_SESSION)
    assert rc.lifecycle_state.positions["GME"]["qty"] == 60


@pytest.mark.asyncio
async def test_tick_reconcile_leaves_pending_exits_to_the_daemon(tmp_path: Path) -> None:
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    await broker.submit_buy("GME", 100, 18.0, _IN_SESSION)
    rc = _rc(tmp_path, broker, JsonStateStore(tmp_path / "state.json"))
    await rc.setup()
    meta = rc.lifecycle_state.positions["GME"]
    meta["pending_exits"] = ["sell-1"]
    meta["pending_action"] = "halve"
    meta["pending_qty"] = 50
    await broker.submit_sell("GME", 50, 18.0, _IN_SESSION)  # the halve filled in between
    # The daemon's own reconcile (get_position_qty) handles it; the tick-level
    # reconcile must not race it or double-count.
    rc.broker.fetch_quote = AsyncMock(  # type: ignore[union-attr]
        return_value=__import__("squeeze_hunter.broker.base", fromlist=["Quote"]).Quote(
            ticker="GME", bid=18.0, ask=18.0, last=18.0, timestamp_ns=0
        )
    )
    await rc.tick(now=_IN_SESSION)
    assert rc.lifecycle_state.positions["GME"]["qty"] == 50
    assert rc.lifecycle_state.positions["GME"].get("halved") is True


@pytest.mark.asyncio
async def test_eod_full_reconcile_alerts_on_drift(tmp_path: Path) -> None:
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    await broker.submit_buy("GME", 100, 18.0, _IN_SESSION)
    rc = _rc(tmp_path, broker, JsonStateStore(tmp_path / "state.json"))
    await rc.setup()
    rc.alerts = AlertSender(telegram_bot_token="t", telegram_chat_id="c", slack_webhook_url=None)
    rc.alerts._send_telegram = AsyncMock()  # type: ignore[method-assign]
    await broker.submit_sell("GME", 100, 18.0, _IN_SESSION)  # flat at the broker
    await rc.eod_close(now=datetime(2026, 5, 14, 20, 30, tzinfo=UTC))
    assert "GME" not in rc.lifecycle_state.positions
    rc.alerts._send_telegram.assert_awaited_once()
    assert "reconcil" in rc.alerts._send_telegram.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_no_store_configured_means_no_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rc = RuntimeContext(cache=_cache(tmp_path), settings=_settings(), tickers=["GME"], mode="sim")
    await rc.setup()
    await rc.eod_close(now=datetime(2026, 5, 14, 20, 30, tzinfo=UTC))
    assert not (tmp_path / "data").exists()


def test_position_snapshot_is_a_plain_record() -> None:
    p = PositionSnapshot(ticker="GME", qty=10, avg_cost=18.5)
    assert (p.ticker, p.qty, p.avg_cost) == ("GME", 10, 18.5)
