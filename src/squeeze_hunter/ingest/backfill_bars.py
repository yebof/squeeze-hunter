"""Backfill EOD bars from yfinance into parquet cache."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.yahoo import YahooProvider
from squeeze_hunter.logging_setup import get_logger

log = get_logger("ingest.bars")


async def backfill_bars_for_ticker(
    ticker: str, start: datetime, end: datetime, cache: ParquetCache
) -> None:
    provider = YahooProvider()
    bars = await provider.fetch_bars(ticker, start, end)
    if not bars:
        # R6.M5: log when yfinance returns empty so a rate-limited or
        # delisted ticker is visible in structured logs. Otherwise a
        # silent skip during a large backfill leaves no trail.
        log.warning(
            "backfill_bars_empty",
            ticker=ticker,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        return
    df = pd.DataFrame(
        [
            {
                "ticker": b.ticker,
                "ts": b.ts,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
    )
    cache.append_partition("bars", ticker, df, dedup_keys=["ticker", "ts"])


async def backfill_bars_for_universe(
    tickers: list[str], start: datetime, end: datetime, cache: ParquetCache
) -> None:
    for t in tickers:
        await backfill_bars_for_ticker(t, start, end, cache)
