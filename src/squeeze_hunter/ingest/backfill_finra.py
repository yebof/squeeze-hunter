"""Backfill historical FINRA short-interest into parquet cache, with float merge."""

from __future__ import annotations

from datetime import date

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.finra import FinraProvider
from squeeze_hunter.data.providers.yahoo import YahooProvider
from squeeze_hunter.logging_setup import get_logger

log = get_logger("ingest.finra")


async def backfill_finra(tickers: list[str], cache: ParquetCache) -> None:
    finra = FinraProvider()
    yahoo = YahooProvider()
    rows: list[dict] = []
    float_cache: dict[str, int | None] = {}

    for t in tickers:
        si_list = await finra.fetch_short_interest(t, since=date(2018, 1, 1))
        if not si_list:
            continue
        # Look up float once per ticker; cache to avoid duplicate Yahoo calls.
        if t not in float_cache:
            try:
                float_cache[t] = await yahoo.get_float_shares(t)
            except (ConnectionError, TimeoutError, OSError, KeyError, ValueError) as e:
                # R8.M11: narrow per CLAUDE.md. KeyError/ValueError cover
                # yfinance's "no float in info" path; transient errors cover
                # network. AttributeError must propagate (real provider bug).
                log.warning("yahoo_float_failed", ticker=t, err=str(e), err_type=type(e).__name__)
                float_cache[t] = None
        float_shares = float_cache[t]

        for si in si_list:
            si_pct_float = (si.si_shares / float_shares) if float_shares else 0.0
            rows.append(
                {
                    "ticker": si.ticker,
                    "settlement_date": si.settlement_date,
                    "si_shares": si.si_shares,
                    "si_pct_float": si_pct_float,
                    "avg_daily_volume_20d": si.avg_daily_volume_20d,
                }
            )
    if not rows:
        return
    cache.append_partition(
        "short_interest", "all", pd.DataFrame(rows), dedup_keys=["ticker", "settlement_date"]
    )
