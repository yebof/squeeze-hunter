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
