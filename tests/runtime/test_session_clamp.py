"""Round-12 runtime regressions.

1. Broker-outage / data-stale killswitch arms must measure outage time WITHIN
   the session. Ticks outside 09:30-16:00 ET never touch telemetry, so the
   heartbeat froze at ~15:59 ET; one transient health() failure at the next
   open read as a 17h+ outage and locked the system out for 7 days.
2. While the killswitch cooldown suppresses candidates, held positions must
   still get their current_score refreshed — otherwise the signal-decay stops
   are dead for the entire cooldown.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.runtime import RuntimeContext


def _seed(cache: ParquetCache) -> None:
    base = datetime(2026, 5, 14, tzinfo=UTC)
    rows = [
        {
            "ticker": "GME",
            "ts": base + timedelta(days=i),
            "open": 18.0,
            "high": 18.5,
            "low": 17.5,
            "close": 18.0,
            "volume": 1_000_000,
        }
        for i in range(30)
    ]
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


async def _runtime(tmp_path: Path) -> RuntimeContext:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {"f6_bollinger_breakout": 1.0, "f7_volume_spike": 1.0}
    rc = RuntimeContext(cache=cache, settings=settings, tickers=["GME"], mode="sim")
    await rc.setup()
    return rc


@pytest.mark.asyncio
async def test_broker_outage_arm_ignores_the_overnight_gap(tmp_path: Path) -> None:
    rc = await _runtime(tmp_path)
    assert rc.broker is not None
    yesterday_close = datetime(2026, 5, 13, 19, 59, tzinfo=UTC)  # Wed 15:59 ET
    rc.telemetry.record_broker_heartbeat(yesterday_close)
    rc.telemetry.record_data_freshness("ibkr_quotes", yesterday_close)
    rc.broker.health = AsyncMock(side_effect=ConnectionError("tws down"))  # type: ignore[method-assign]

    # Thu 09:31 ET: one failed health() at the open is a 60 s outage, not 17 h.
    await rc.tick(now=datetime(2026, 5, 14, 13, 31, tzinfo=UTC))
    assert not rc.kill_switch_active, rc._kill_reason

    # Still down at 09:36 ET: 6 min in-session outage ≥ 300 s → trip.
    await rc.tick(now=datetime(2026, 5, 14, 13, 36, tzinfo=UTC))
    assert rc.kill_switch_active
    assert rc._kill_reason == "broker_outage"


@pytest.mark.asyncio
async def test_nightly_scan_refreshes_current_score_during_killswitch_cooldown(
    tmp_path: Path,
) -> None:
    rc = await _runtime(tmp_path)
    rc.lifecycle_state.positions["GME"] = {
        "qty": 100,
        "entry_price": 100.0,
        "peak_price": 100.0,
        "entry_score": 10.0,
        "current_score": 10.0,
        "bars_held": 0,
        "setup_type": "CAR",
    }
    rc.kill_switch_active = True
    rc._kill_reason = "monthly_drawdown"

    await rc.nightly_scan(now=datetime(2026, 6, 10, 2, 0, tzinfo=UTC))

    assert rc.last_candidates is not None
    assert rc.last_candidates.empty
    assert rc.lifecycle_state.positions["GME"]["current_score"] != 10.0
