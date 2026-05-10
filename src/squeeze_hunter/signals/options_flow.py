"""Signal f5 — ATM call OI 7-day velocity.

raw_value = (oi_today_atm - oi_7d_ago_atm) / max(oi_7d_ago_atm, baseline)

ATM band: strikes within 5% of spot.

The 7-day-ago snapshot is read from a separate archive (the daily-scan
job persists option chains under data/options/<ticker>/<date>.parquet).
The `_archive` keyword is a test seam — production reads from the cache.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.data.schema import OptionChain
from squeeze_hunter.signals.base import Factor


def _atm_call_oi(chain: OptionChain, window_pct: float = 0.05) -> int:
    return chain.total_call_oi(near_money_window_pct=window_pct)


async def compute_call_oi_velocity(
    tickers: list[str],
    provider: DataProvider,
    clock: datetime,
    *,
    _archive: dict[tuple[str, date], OptionChain] | None = None,
) -> Factor:
    archive = _archive or {}
    rows = []
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
        prior_date = (clock - timedelta(days=7)).date()
        prior = archive.get((t, prior_date))
        if prior is None or not prior.quotes:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        oi_prior = _atm_call_oi(prior)
        baseline = max(oi_prior, 100)
        raw = (oi_today - oi_prior) / baseline
        rows.append({"ticker": t, "raw_value": raw})
    return Factor(name="f5_call_oi_velocity", as_of=clock, values=pd.DataFrame(rows))
