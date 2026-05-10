"""Backfill earnings calendar into parquet cache."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.finnhub import FinnhubProvider


async def backfill_earnings(tickers: list[str], cache: ParquetCache) -> None:
    api_key = os.environ.get("FINNHUB_KEY", "")
    provider = FinnhubProvider(api_key=api_key)
    rows: list[dict] = []
    for t in tickers:
        events = await provider.fetch_earnings(t, since=date(2018, 1, 1))
        for e in events:
            rows.append(
                {
                    "ticker": e.ticker,
                    "report_at": e.report_at,
                    "actual_eps": e.actual_eps,
                    "estimate_eps": e.estimate_eps,
                }
            )
    if not rows:
        return
    cache.append_partition(
        "earnings", "all", pd.DataFrame(rows), dedup_keys=["ticker", "report_at"]
    )
