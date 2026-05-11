"""Compute all 7 factors and stack into a long-format dataframe with z-scores."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.signals.base import Factor
from squeeze_hunter.signals.earnings_reaction import compute_earnings_reaction
from squeeze_hunter.signals.normalize import cross_sectional_z
from squeeze_hunter.signals.options_flow import compute_call_oi_velocity
from squeeze_hunter.signals.sentiment import compute_wsb_sentiment
from squeeze_hunter.signals.short_interest import compute_days_to_cover, compute_si_pct_float
from squeeze_hunter.signals.technicals import compute_bollinger_breakout, compute_volume_spike

log = get_logger("signals.compute")

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
    """Compute all 7 factors concurrently and stack into a long-format dataframe.

    R6.I5: uses return_exceptions=True so a transient failure in ONE signal
    (e.g., a network error in fetch_short_interest) doesn't take down the
    entire scan. Failed factors are logged and skipped; the remaining
    factors still contribute. The combiner downstream warns when factors
    are unmapped, and raises if NONE map.
    """
    factor_fns = (
        ("f1_si_pct", compute_si_pct_float),
        ("f2_days_to_cover", compute_days_to_cover),
        ("f3_earnings_reaction", compute_earnings_reaction),
        ("f4_wsb_mention", compute_wsb_sentiment),
        ("f5_call_oi_velocity", compute_call_oi_velocity),
        ("f6_bollinger_breakout", compute_bollinger_breakout),
        ("f7_volume_spike", compute_volume_spike),
    )
    results = await asyncio.gather(
        *(fn(tickers, provider, clock) for _, fn in factor_fns),
        return_exceptions=True,
    )
    frames = []
    for (name, _fn), result in zip(factor_fns, results, strict=True):
        if isinstance(result, BaseException):
            log.warning(
                "factor_compute_failed",
                factor=name,
                err=str(result),
                err_type=type(result).__name__,
            )
            continue
        f: Factor = result  # type: ignore[assignment]
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
