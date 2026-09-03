"""Backfill earnings calendar into parquet cache."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.finnhub import FinnhubProvider
from squeeze_hunter.logging_setup import get_logger

_log = get_logger("ingest.earnings")


async def backfill_earnings(tickers: list[str], cache: ParquetCache) -> None:
    api_key = os.environ.get("FINNHUB_KEY", "")
    if not api_key:
        # Round-12: FinnhubProvider silently returns no events without a key,
        # so this used to exit 0 having written nothing — and f3 (earnings
        # reaction, weight 2.0) was dead for every ticker with no hint why.
        raise ValueError(
            "FINNHUB_KEY is not set: refusing to run a no-op earnings backfill "
            "(the `earnings` partition would stay empty and f3 would be dead)"
        )
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
        _log.warning("earnings_backfill_no_rows", n_tickers=len(tickers))
        return
    cache.append_partition(
        "earnings", "all", pd.DataFrame(rows), dedup_keys=["ticker", "report_at"]
    )
