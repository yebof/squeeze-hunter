"""Gate-context assembly from the parquet cache — shared by backtest and live.

P1: the backtest runner used to build ADV20 / price-floor / earnings-proximity
inputs inline. The live premarket path needs the identical inputs, so they
are built here from the same cache and the same clock semantics.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.risk.gates import GateContext
from squeeze_hunter.signals.earnings_reaction import _trading_days_between


def earnings_within_days(
    cache: ParquetCache, tickers: list[str], as_of: datetime, *, days: int = 3
) -> dict[str, bool]:
    """I10 / R7.M1: a known future report date is public information, so this
    reads the earnings cache directly (no look-ahead guard) and counts NYSE
    trading days, not calendar days."""
    out = {t: False for t in tickers}
    df = cache.read_partition("earnings", "all")
    if df.empty:
        return out
    df["report_at"] = pd.to_datetime(df["report_at"], utc=True)
    for t in tickers:
        for _, row in df[df["ticker"] == t].iterrows():
            if 0 <= _trading_days_between(as_of, row["report_at"]) <= days:
                out[t] = True
                break
    return out


async def build_gate_context(
    provider: DataProvider,
    cache: ParquetCache,
    tickers: list[str],
    as_of: datetime,
    settings: Settings,
    *,
    kill_switch_active: bool,
) -> GateContext:
    """Round-12: real liquidity and price inputs. ADV20$ is the mean
    close*volume of the 20 bars BEFORE the latest session (a spike day must
    not certify its own liquidity — Round-13 requires at least one prior bar);
    the universe is every ticker with a bar for `as_of`'s date whose close
    clears settings.universe.min_price. Listing age (365), halts (none) and
    pairwise correlations (0) remain placeholders: no data source yet."""
    adv20: dict[str, float] = {}
    universe: set[str] = set()
    for t in tickers:
        try:
            bars = await provider.fetch_bars(t, as_of - timedelta(days=40), as_of)
        except LookupError:
            continue
        if not bars or bars[-1].ts.date() != as_of.date():
            continue
        prior = bars[:-1][-20:]
        if not prior:
            continue
        adv20[t] = sum(b.close * b.volume for b in prior) / len(prior)
        if bars[-1].close >= settings.universe.min_price:
            universe.add(t)
    return GateContext(
        as_of=as_of,
        kill_switch_active=kill_switch_active,
        adv20_dollar_volume_by_ticker=adv20,
        days_listed_by_ticker={t: 365 for t in tickers},
        halted_tickers=frozenset(),
        universe_tickers=frozenset(universe),
        earnings_within_3_days=earnings_within_days(cache, tickers, as_of),
        portfolio_correlations={t: 0.0 for t in tickers},
    )
