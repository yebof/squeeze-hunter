"""Compute all 7 factors and stack into a long-format dataframe with z-scores."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.signals.earnings_reaction import compute_earnings_reaction
from squeeze_hunter.signals.normalize import cross_sectional_z
from squeeze_hunter.signals.options_flow import compute_call_oi_velocity
from squeeze_hunter.signals.sentiment import compute_wsb_sentiment
from squeeze_hunter.signals.short_interest import compute_days_to_cover, compute_si_pct_float
from squeeze_hunter.signals.technicals import compute_bollinger_breakout, compute_volume_spike

FACTOR_NAMES = (
    "f1_si_pct",
    "f2_days_to_cover",
    "f3_earnings_reaction",
    "f4_wsb_mention",
    "f5_call_oi_velocity",
    "f6_bollinger_breakout",
    "f7_volume_spike",
)


async def compute_all_factors(
    tickers: list[str], provider: DataProvider, clock: datetime
) -> pd.DataFrame:
    factors = await asyncio.gather(
        compute_si_pct_float(tickers, provider, clock),
        compute_days_to_cover(tickers, provider, clock),
        compute_earnings_reaction(tickers, provider, clock),
        compute_wsb_sentiment(tickers, provider, clock),
        compute_call_oi_velocity(tickers, provider, clock),
        compute_bollinger_breakout(tickers, provider, clock),
        compute_volume_spike(tickers, provider, clock),
    )
    frames = []
    for f in factors:
        if f.values.empty:
            continue
        v = f.values.copy()
        v["factor_name"] = f.name
        v["z_score"] = cross_sectional_z(v["raw_value"])
        frames.append(v[["ticker", "factor_name", "raw_value", "z_score"]])
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["ticker", "factor_name", "raw_value", "z_score"])
    )
