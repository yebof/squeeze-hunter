from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.runtime import RuntimeContext


def _seed(cache: ParquetCache) -> None:
    base = datetime(2026, 5, 14, tzinfo=UTC)
    rows = []
    for i in range(30):
        rows.append(
            {
                "ticker": "GME",
                "ts": base + timedelta(days=i),
                "open": 18.0,
                "high": 18.5,
                "low": 17.5,
                "close": 18.0,
                "volume": 1_000_000,
            }
        )
    cache.write_partition("bars", "GME", pd.DataFrame(rows))
    cache.write_partition(
        "short_interest",
        "all",
        pd.DataFrame(
            columns=[
                "ticker",
                "settlement_date",
                "si_shares",
                "si_pct_float",
                "avg_daily_volume_20d",
            ]
        ),
    )
    cache.write_partition(
        "earnings",
        "all",
        pd.DataFrame(columns=["ticker", "report_at", "actual_eps", "estimate_eps"]),
    )


@pytest.mark.asyncio
async def test_runtime_one_tick_no_signals(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {"f1_si_pct": 1.0, "f7_volume_spike": 1.0}
    rc = RuntimeContext(
        cache=cache,
        settings=settings,
        tickers=["GME"],
        mode="sim",
        broker=None,
    )
    await rc.setup()
    await rc.tick(now=datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    assert rc.metrics_registry is not None
    assert rc.kill_switch_active is False


@pytest.mark.asyncio
async def test_tick_skipped_outside_us_regular_session(tmp_path: Path) -> None:
    """R3.2 regression: intraday tick must NOT run outside Mon-Fri 09:30-16:00 ET.

    Before the fix: stops were evaluated 24/7 and after-hours triggers
    submitted MarketOrders into AH liquidity (3-8% adverse slippage typical).
    """
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(cache=cache, settings=Settings(), tickers=["GME"], mode="sim")
    await rc.setup()

    # Seed a clearly-tripping killswitch state — if tick runs, killswitch flips.
    rc.telemetry.record_equity(datetime(2026, 5, 11, 14, 0, tzinfo=UTC), 100_000.0)
    rc.telemetry.record_equity(datetime(2026, 5, 11, 15, 0, tzinfo=UTC), 80_000.0)  # -20%
    rc.telemetry.record_broker_heartbeat(datetime(2026, 5, 11, 15, 0, tzinfo=UTC))
    rc.telemetry.record_data_freshness("ibkr_quotes", datetime(2026, 5, 11, 15, 0, tzinfo=UTC))

    # Saturday — outside session. Tick must return without flipping killswitch.
    await rc.tick(now=datetime(2026, 5, 16, 14, 0, tzinfo=UTC))
    assert rc.kill_switch_active is False, "tick on Saturday should skip work"

    # After regular close (17:00 ET = 21:00 UTC in EDT). Still outside session.
    await rc.tick(now=datetime(2026, 5, 11, 21, 0, tzinfo=UTC))
    assert rc.kill_switch_active is False, "tick after 16:00 ET should skip work"

    # Before market open (08:00 ET = 12:00 UTC in EDT). Outside session.
    await rc.tick(now=datetime(2026, 5, 11, 12, 0, tzinfo=UTC))
    assert rc.kill_switch_active is False, "tick before 09:30 ET should skip work"

    # In session — killswitch should now actually trip.
    await rc.tick(now=datetime(2026, 5, 11, 14, 0, tzinfo=UTC))  # 10:00 ET Mon
    assert rc.kill_switch_active is True


@pytest.mark.asyncio
async def test_tick_clears_telemetry_position_marks_on_exit(tmp_path: Path) -> None:
    """R3.1 regression: when a position is exited during tick, its telemetry
    mark must be cleared. Otherwise worst_position_gap_pct keeps reading the
    last adverse mark forever and permanently arms the gap-through-stop
    killswitch.

    Before the fix: one bad overnight gap (e.g., -30%) would exit the position
    via hard stop but the (entry, mark) tuple stayed in position_marks,
    causing kill_switch_active=True for the lifetime of the process.

    Test strategy: install a broker that returns a low quote (forcing hard
    stop), let manage_positions exit the position, then verify telemetry is
    cleared.
    """
    from unittest.mock import AsyncMock, MagicMock

    from squeeze_hunter.broker.base import BrokerHealth, BrokerOrder, Quote

    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(cache=cache, settings=Settings(), tickers=["GME"], mode="sim")

    # Use a mock broker so we can force a hard-stop quote.
    mock_broker = MagicMock()
    mock_broker.name = "mock"
    mock_broker.connect = AsyncMock()
    mock_broker.disconnect = AsyncMock()
    mock_broker.health = AsyncMock(
        return_value=BrokerHealth(connected=True, last_ping_ms=0, account="mock"),
    )
    # Quote far below entry → hard stop fires
    mock_broker.fetch_quote = AsyncMock(
        return_value=Quote(ticker="GME", bid=70.0, ask=70.05, last=70.0, timestamp_ns=0),
    )
    mock_broker.submit_sell = AsyncMock(
        return_value=BrokerOrder(
            broker_order_id="x1",
            ticker="GME",
            side="sell",
            qty=100,
            limit_price=None,
            status="filled",
            filled_qty=100,
            avg_fill_price=70.0,
        ),
    )
    rc.broker = mock_broker
    await rc.setup()

    # Open position in lifecycle and seed a stale telemetry mark
    rc.lifecycle_state.positions["GME"] = {
        "qty": 100,
        "entry_price": 100.0,
        "peak_price": 110.0,
        "entry_score": 10.0,
        "current_score": 10.0,
        "bars_held": 5,
        "setup_type": "CAR",
    }
    rc.telemetry.record_position("GME", entry_price=100.0, mark_price=70.0)
    assert rc.telemetry.worst_position_gap_pct() == pytest.approx(-0.30)

    # Run tick during market hours. The quote of 70.0 vs entry 100.0 = -30%
    # → hard stop at -12% fires → manage_positions exits the position.
    await rc.tick(now=datetime(2026, 5, 11, 14, 0, tzinfo=UTC))  # Mon 10:00 ET

    # The position is gone from lifecycle...
    assert "GME" not in rc.lifecycle_state.positions
    # ...and its stale telemetry mark must be cleared too.
    assert "GME" not in rc.telemetry.position_marks
    assert rc.telemetry.worst_position_gap_pct() == 0.0


@pytest.mark.asyncio
async def test_setup_connect_times_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """R3.3 regression: setup() bounds broker.connect with asyncio.wait_for.
    Previously a hung connectAsync froze the process indefinitely with no
    diagnostic.
    """
    import asyncio as _asyncio

    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(cache=cache, settings=Settings(), tickers=["GME"], mode="paper")

    # Inject a broker whose connect() never returns
    class HangingBroker:
        name = "hanging"
        port = 7497

        async def connect(self) -> None:
            await _asyncio.sleep(10_000)

        async def disconnect(self) -> None:
            pass

        async def health(self):
            from squeeze_hunter.broker.base import BrokerHealth

            return BrokerHealth(connected=False, last_ping_ms=0, account="hanging")

    rc.broker = HangingBroker()  # type: ignore[assignment]
    # Already-set broker skips the connect() call. So instead patch the paper
    # branch by setting broker=None and the import path.
    rc.broker = None

    from squeeze_hunter.broker import paper as paper_module

    original_paper_broker = paper_module.PaperBroker

    def _hanging_factory(*args, **kwargs):
        b = HangingBroker()
        return b

    monkeypatch.setattr(paper_module, "PaperBroker", _hanging_factory)

    # setup() with a 0.5s timeout should raise TimeoutError.
    with pytest.raises(TimeoutError):
        await rc.setup(connect_timeout_s=0.5)

    # restore
    monkeypatch.setattr(paper_module, "PaperBroker", original_paper_broker)


@pytest.mark.asyncio
async def test_runtime_killswitch_uses_real_telemetry(tmp_path: Path) -> None:
    """C1 regression: tick() must feed real telemetry to evaluate_killswitch,
    not all-zero placeholders.
    """
    cache = ParquetCache(root=tmp_path)
    _seed(cache)  # existing helper in this file
    settings = Settings()
    rc = RuntimeContext(
        cache=cache,
        settings=settings,
        tickers=["GME"],
        mode="sim",
    )
    await rc.setup()
    # Simulate a -15% drawdown in the telemetry. Use weekday timestamps within
    # US regular session (10:00 ET = 14:00 UTC during EDT) — the R3.2 market
    # hours guard now skips tick() outside the session.
    day0 = datetime(2026, 5, 11, 14, 0, tzinfo=UTC)  # Mon
    day1 = datetime(2026, 5, 12, 14, 0, tzinfo=UTC)  # Tue
    day2 = datetime(2026, 5, 13, 14, 0, tzinfo=UTC)  # Wed
    rc.telemetry.record_equity(day0, 100_000.0)
    rc.telemetry.record_equity(day1, 110_000.0)
    rc.telemetry.record_equity(day2, 93_500.0)  # -15% from peak
    rc.telemetry.record_broker_heartbeat(day2)
    rc.telemetry.record_data_freshness("ibkr_quotes", day2)

    await rc.tick(now=day2)
    assert rc.kill_switch_active is True
    assert rc._kill_reason == "monthly_drawdown"
