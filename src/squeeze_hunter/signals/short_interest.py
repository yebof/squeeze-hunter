"""Signals f1 (SI % of float) and f2 (Days-to-cover)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.signals.base import Factor


async def compute_si_pct_float(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    rows = []
    for t in tickers:
        si_list = await provider.fetch_short_interest(t)
        if not si_list:
            continue
        latest = max(si_list, key=lambda x: x.settlement_date)
        if latest.si_pct_float <= 0.0:
            # No float data for this ticker — exclude from f1 rather than emit 0,
            # which would pollute the cross-sectional z-score for other tickers.
            # The ticker still participates in f2-f7.
            continue
        rows.append({"ticker": t, "raw_value": latest.si_pct_float})
    return Factor(name="f1_si_pct", as_of=clock, values=pd.DataFrame(rows))


async def compute_days_to_cover(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    rows = []
    for t in tickers:
        si_list = await provider.fetch_short_interest(t)
        if not si_list:
            continue
        latest = max(si_list, key=lambda x: x.settlement_date)
        rows.append({"ticker": t, "raw_value": float(latest.days_to_cover)})
    return Factor(name="f2_days_to_cover", as_of=clock, values=pd.DataFrame(rows))
