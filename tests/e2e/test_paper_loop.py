from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.execution.lifecycle import LifecycleState
from squeeze_hunter.runtime import RuntimeContext


@pytest.mark.asyncio
async def test_runtime_three_ticks_no_crash(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    base = datetime(2026, 5, 14, tzinfo=UTC)
    bars = [
        {
            "ticker": "GME",
            "ts": base + timedelta(days=i),
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1_000_000,
        }
        for i in range(30)
    ]
    cache.write_partition("bars", "GME", pd.DataFrame(bars))
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

    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    rc.lifecycle_state = LifecycleState(
        positions={
            "GME": {
                "qty": 100,
                "entry_price": 100.0,
                "peak_price": 100.0,
                "entry_score": 10.0,
                "current_score": 10.0,
                "bars_held": 1,
                "setup_type": "CAR",
            }
        }
    )
    await rc.setup()
    # Three ticks at increasing time. SimulatorBroker has no fetch_quote, so
    # lifecycle logs a warning per ticker and continues — no crash.
    for offset in (0, 60, 120):
        await rc.tick(now=base + timedelta(days=29, seconds=offset))
    await rc.shutdown()
