"""Signal f4 — Reddit WSB mention z-score (24h vs 30d baseline)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.signals.base import Factor


async def compute_wsb_sentiment(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    rows = []
    for t in tickers:
        m = await provider.fetch_sentiment(t, clock)
        if m is None:
            rows.append({"ticker": t, "raw_value": float("nan")})
            continue
        rows.append({"ticker": t, "raw_value": float(m.z_score)})
    return Factor(name="f4_wsb_mention", as_of=clock, values=pd.DataFrame(rows))
