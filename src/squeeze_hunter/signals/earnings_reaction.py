"""Signal f3 — earnings reaction.

raw_value = sign(gap) * |gap| * log(1 + max(0, volume_ratio - 1))

where:
  gap            = (close_first_session_after - close_session_before) / close_session_before
  volume_ratio   = post_volume / mean(pre_20d_volume)

Captures fundamental catalysts via *price reaction* without paid news/NLP.
Only earnings within the past 5 trading days contribute; older ones decay to 0.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from math import log

import numpy as np
import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.signals.base import Factor


def _trading_days_between(d1: datetime, d2: datetime) -> int:
    """Count trading days from d1 to d2 (exclusive of d2 start, inclusive of d2 end).

    Uses numpy.busday_count which counts Mon-Fri business days.
    Returns a positive integer when d2 > d1.
    """
    return int(
        np.busday_count(
            d1.date() if hasattr(d1, "date") else d1,
            d2.date() if hasattr(d2, "date") else d2,
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
