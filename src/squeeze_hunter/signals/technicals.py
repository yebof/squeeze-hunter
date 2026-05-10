"""Signals f6 (Bollinger breakout) and f7 (Volume spike vs ADV20)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.signals.base import Factor


async def compute_bollinger_breakout(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    """raw = max(0, (close - upper_band) / std) when band-width is in lowest 25% of last 60d.

    Captures "squeeze then breakout": tight Bollinger band followed by close > upper band.
    """
    rows = []
    for t in tickers:
        bars = await provider.fetch_bars(t, clock - timedelta(days=120), clock)
        if len(bars) < 25:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        df = pd.DataFrame({"close": [b.close for b in bars]}).iloc[-60:]
        # Compute bands using PRE-TODAY data only, so today's gap-up doesn't
        # inflate sd20 and suppress the breakout signal (B3 fix).
        prev = df["close"].iloc[:-1]
        ma20 = prev.rolling(20).mean()
        sd20 = prev.rolling(20).std(ddof=0)
        upper = ma20 + 2 * sd20
        band_width = (upper - (ma20 - 2 * sd20)) / ma20
        bw_hist = band_width.dropna()
        bw_quartile = bw_hist.quantile(0.25) if len(bw_hist) > 0 else np.nan
        bw_min_recent = bw_hist.tail(10).min() if len(bw_hist) >= 10 else np.nan
        last_close = df["close"].iloc[-1]
        last_upper = upper.iloc[-1]
        last_sd = sd20.iloc[-1]
        squeezed = (
            not np.isnan(bw_min_recent)
            and not np.isnan(bw_quartile)
            and bw_min_recent <= bw_quartile
        )
        breakout = max(0.0, (last_close - last_upper) / max(last_sd, 1e-9))
        raw = breakout if squeezed else 0.0
        rows.append({"ticker": t, "raw_value": float(raw)})
    return Factor(name="f6_bollinger_breakout", as_of=clock, values=pd.DataFrame(rows))


async def compute_volume_spike(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> Factor:
    """raw = today_volume / mean(volume_20d_excl_today)."""
    rows = []
    for t in tickers:
        bars = await provider.fetch_bars(t, clock - timedelta(days=40), clock)
        if len(bars) < 21:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        bars = sorted(bars, key=lambda b: b.ts)
        last20 = bars[-21:-1]
        today = bars[-1]
        adv = sum(b.volume for b in last20) / 20
        if adv <= 0:
            rows.append({"ticker": t, "raw_value": 0.0})
            continue
        rows.append({"ticker": t, "raw_value": today.volume / adv})
    return Factor(name="f7_volume_spike", as_of=clock, values=pd.DataFrame(rows))
