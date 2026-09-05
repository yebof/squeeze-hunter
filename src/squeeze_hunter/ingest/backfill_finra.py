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
    splits_cache: dict[str, list[tuple[date, float]]] = {}

    # CDX2-P2: ONE pass over the FINRA files for the whole universe, not one
    # full re-download per ticker. fetch_short_interest_bulk GETs each monthly
    # report exactly once and indexes the requested tickers.
    si_by_ticker = await finra.fetch_short_interest_bulk(tickers, since=date(2018, 1, 1))

    for t in tickers:
        si_list = si_by_ticker.get(t, [])
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
        # Round-13: FINRA reports share counts as of the settlement date and
        # never restates them, while Yahoo's float is TODAY's. A 10M short
        # before a 4:1 split is 40M of today's shares; dividing the raw count
        # by today's float understated pre-split f1 by 4x (GME) and overstated
        # it 10x across AMC's 1:10 reverse split. Scale each record by every
        # split that happened AFTER its settlement date. si_shares itself is
        # stored as reported (days_to_cover divides it by an unadjusted ADV).
        if t not in splits_cache:
            try:
                splits_cache[t] = await yahoo.get_split_ratios(t)
            except (ConnectionError, TimeoutError, OSError, KeyError, ValueError) as e:
                log.warning("yahoo_splits_failed", ticker=t, err=str(e), err_type=type(e).__name__)
                splits_cache[t] = []
        splits = splits_cache[t]

        for si in si_list:
            adjust = 1.0
            for split_date, ratio in splits:
                if split_date > si.settlement_date:
                    adjust *= ratio
            adjusted_shares = si.si_shares * adjust
            si_pct_float = (adjusted_shares / float_shares) if float_shares else 0.0
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
        log.warning("finra_backfill_no_rows", n_tickers=len(tickers))
        return
    cache.append_partition(
        "short_interest", "all", pd.DataFrame(rows), dedup_keys=["ticker", "settlement_date"]
    )
