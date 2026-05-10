"""Backfill historical FINRA short-interest into parquet cache."""

from __future__ import annotations

from datetime import date

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.finra import FinraProvider


async def backfill_finra(tickers: list[str], cache: ParquetCache) -> None:
    provider = FinraProvider()
    rows: list[dict] = []
    for t in tickers:
        si_list = await provider.fetch_short_interest(t, since=date(2018, 1, 1))
        for si in si_list:
            rows.append(
                {
                    "ticker": si.ticker,
                    "settlement_date": si.settlement_date,
                    "si_shares": si.si_shares,
                    "si_pct_float": si.si_pct_float,
                    "avg_daily_volume_20d": si.avg_daily_volume_20d,
                }
            )
    if not rows:
        return
    cache.dedup_keys = ["ticker", "settlement_date"]
    cache.append_partition("short_interest", "all", pd.DataFrame(rows))
