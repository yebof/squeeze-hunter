"""Signal f3 — earnings reaction.

raw_value = sign(gap) * |gap| * log(1 + max(0, volume_ratio - 1))

where:
  gap            = (close_first_session_after - close_session_before) / close_session_before
  volume_ratio   = post_volume / mean(pre_20d_volume)

Captures fundamental catalysts via *price reaction* without paid news/NLP.
Only earnings within the past 5 trading days contribute; older ones decay to 0.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from math import log

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.signals.base import Factor

# R8: cache the trading holidays once. np.busday_count without a holidays
# argument counts Mon-Fri only and treats Christmas / Thanksgiving / etc. as
# trading days.
# Round-13: the calendar is the real NYSE one (pandas_market_calendars), not
# the US federal calendar. The federal approximation iterated Good Friday
# (NYSE closed → an equity point with no marks every year) and skipped
# Columbus Day and Veterans Day (NYSE open → never scanned, never
# stop-checked, missing from the equity curve). Every consumer — the
# backtest day loop, the live time-stop counter, the f3 / f5 windows, the
# FINRA publication lag and captured-events — shares this list.
_US_HOLIDAYS_CACHE: list[date] | None = None


def _us_business_holidays() -> list[date]:
    """Weekday NYSE closures 2018-2030 (name kept for the existing callers)."""
    global _US_HOLIDAYS_CACHE
    if _US_HOLIDAYS_CACHE is None:
        nyse = mcal.get_calendar("NYSE")
        # Weekdays that are NOT sessions on the exchange schedule: regular
        # holidays plus special closures (days of mourning, etc.).
        sessions = {ts.date() for ts in nyse.schedule("2018-01-01", "2030-12-31").index}
        weekdays = pd.bdate_range("2018-01-01", "2030-12-31")
        _US_HOLIDAYS_CACHE = [ts.date() for ts in weekdays if ts.date() not in sessions]
    return _US_HOLIDAYS_CACHE


def _trading_days_between(d1: datetime, d2: datetime) -> int:
    """Count trading days from d1 to d2 (exclusive of d2 start, inclusive of d2 end).

    Uses numpy.busday_count with US federal holidays excluded — Christmas,
    Thanksgiving, July 4th, etc. no longer count as trading days. Mon-Fri
    only otherwise.

    Returns a positive integer when d2 > d1.
    """
    holidays_dt64 = np.array(_us_business_holidays(), dtype="datetime64[D]")
    return int(
        np.busday_count(
            d1.date() if hasattr(d1, "date") else d1,
            d2.date() if hasattr(d2, "date") else d2,
            holidays=holidays_dt64,
        )
    )


async def compute_earnings_reaction(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    rows = []
    horizon_days = 5
    for t in tickers:
        events = await provider.fetch_earnings(t)
        # B4 fix: filter by trading days, not calendar days
        recent = [
            e for e in events if 0 <= _trading_days_between(e.report_at, clock) <= horizon_days
        ]
        if not recent:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        ev = max(recent, key=lambda e: e.report_at)
        bars = await provider.fetch_bars(t, ev.report_at - timedelta(days=30), clock)
        if len(bars) < 21:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        bars.sort(key=lambda b: b.ts)
        # last bar before event:
        pre_event_bars = [b for b in bars if b.ts < ev.report_at]
        post_event_bars = [b for b in bars if b.ts > ev.report_at]
        if not pre_event_bars or not post_event_bars:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        last_pre = pre_event_bars[-1]
        first_post = post_event_bars[0]
        gap = (first_post.close - last_pre.close) / last_pre.close
        last_20 = pre_event_bars[-20:]
        avg_vol = sum(b.volume for b in last_20) / max(len(last_20), 1)
        ratio = first_post.volume / avg_vol if avg_vol > 0 else 1.0
        raw = (1 if gap >= 0 else -1) * abs(gap) * log(1 + max(0.0, ratio - 1.0))
        # decay: linear over horizon (in trading days)
        days_old = _trading_days_between(ev.report_at, clock)
        decay = max(0.0, 1.0 - days_old / horizon_days)
        rows.append({"ticker": t, "raw_value": raw * decay})
    return Factor(name="f3_earnings_reaction", as_of=clock, values=pd.DataFrame(rows))
