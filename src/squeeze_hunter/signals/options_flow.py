"""Signal f5 — ATM call OI 7-day velocity.

raw_value = (oi_today_atm - oi_7d_ago_atm) / max(oi_7d_ago_atm, baseline)

ATM band: strikes within 5% of spot.

The 7-day-ago snapshot is read from a separate archive (the daily-scan
job persists option chains under data/options/<ticker>/<date>.parquet).
The `_archive` keyword is a test seam — production reads from the cache.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.data.schema import OptionChain
from squeeze_hunter.signals.base import Factor
from squeeze_hunter.signals.earnings_reaction import _us_business_holidays


def _atm_call_oi(chain: OptionChain, window_pct: float = 0.05) -> int:
    return chain.total_call_oi(near_money_window_pct=window_pct)


def _trading_days_ago(clock: datetime, n_trading_days: int) -> datetime:
    """R7.C10: subtract `n_trading_days` (Mon-Fri minus US federal holidays)
    from `clock`. Calendar-day arithmetic would land on a non-trading day
    after a holiday week → cache lookup misses → factor zeros for the entire
    universe.
    """
    holidays_dt64 = np.array(_us_business_holidays(), dtype="datetime64[D]")
    target = np.busday_offset(
        clock.date(),
        -n_trading_days,
        roll="backward",
        holidays=holidays_dt64,
    )
    # numpy returns datetime64[D]; convert back to a tz-aware datetime if needed.
    # Timestamp.to_pydatetime() is typed as `datetime | NaTType`, but
    # busday_offset never returns NaT for a valid in-range input — narrow
    # explicitly via assert so the type checker sees a clean datetime.
    ts = pd.Timestamp(target).to_pydatetime()
    assert isinstance(ts, datetime), "busday_offset returned NaT"
    if clock.tzinfo is not None:
        ts = ts.replace(tzinfo=clock.tzinfo)
    return ts


async def compute_call_oi_velocity(
    tickers: list[str],
    provider: DataProvider,
    clock: datetime,
    *,
    _archive: dict[tuple[str, date], OptionChain] | None = None,
) -> Factor:
    """When `_archive` is None, query provider.fetch_option_chain_at for the
    chain 5 trading days ago. The `_archive` parameter is retained as a test seam.

    R7.C10: window is 5 *trading* days, not 7 calendar days. A clock landing
    on Tuesday after a holiday week would otherwise step back to a Tuesday
    with no cached chain → factor returns 0 across the universe.
    """
    rows = []
    prior_dt = _trading_days_ago(clock, n_trading_days=5)
    prior_date = prior_dt.date()
    for t in tickers:
        try:
            today = await provider.fetch_option_chain(t)
        except (NotImplementedError, LookupError):
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        if not today.quotes:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        oi_today = _atm_call_oi(today)

        # Prefer test archive if provided; fall back to provider lookup
        prior: OptionChain | None = None
        if _archive is not None:
            prior = _archive.get((t, prior_date))
        if prior is None or not prior.quotes:
            try:
                # R7.C10: use the trading-day-aligned prior_dt computed above.
                prior = await provider.fetch_option_chain_at(t, prior_dt)
            except (NotImplementedError, LookupError):
                # NotImplementedError: provider doesn't support historical chains.
                # LookupError: chain not found for the requested date.
                # AttributeError is intentionally NOT caught: every concrete
                # provider implements fetch_option_chain_at (it's in the
                # DataProvider Protocol). An AttributeError here is a real bug
                # (wrong object passed, None provider) and must propagate up
                # to tick_safe rather than silently zeroing the signal.
                prior = None
        if prior is None or not prior.quotes:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        oi_prior = _atm_call_oi(prior)
        baseline = max(oi_prior, 100)
        raw = (oi_today - oi_prior) / baseline
        rows.append({"ticker": t, "raw_value": raw})
    return Factor(name="f5_call_oi_velocity", as_of=clock, values=pd.DataFrame(rows))
