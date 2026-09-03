"""Round-12: the FINRA publication lag must count US federal holidays like the
rest of the codebase (time stop, f3 window), not plain Mon-Fri."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock


@pytest.mark.asyncio
async def test_finra_lag_counts_us_federal_holidays(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    cache.write_partition(
        "short_interest",
        "all",
        pd.DataFrame(
            [
                {
                    "ticker": "GME",
                    "settlement_date": date(2024, 5, 24),  # Fri before Memorial Day
                    "si_shares": 1_000_000,
                    "si_pct_float": 0.30,
                    "avg_daily_volume_20d": 100_000,
                }
            ]
        ),
    )
    # 8 Mon-Fri days after 05-24 is 06-05; Memorial Day (05-27) pushes it to 06-06.
    early = BacktestProvider(
        cache=cache,
        clock=Clock(now=datetime(2024, 6, 5, 23, 59, tzinfo=UTC)),
        finra_publication_lag_bdays=8,
    )
    assert await early.fetch_short_interest("GME") == []
    published = BacktestProvider(
        cache=cache,
        clock=Clock(now=datetime(2024, 6, 6, 23, 59, tzinfo=UTC)),
        finra_publication_lag_bdays=8,
    )
    assert len(await published.fetch_short_interest("GME")) == 1
