"""Daily scan orchestrator: universe → factors → score → setup → ranked output."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from squeeze_hunter.config import Settings
from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.score.classifier import classify_setups
from squeeze_hunter.score.combiner import combine
from squeeze_hunter.signals.compute import compute_all_factors


async def run_scan(
    tickers: list[str],
    provider: DataProvider,
    clock: datetime,
    settings: Settings,
) -> pd.DataFrame:
    factors = await compute_all_factors(tickers, provider, clock)
    if factors.empty:
        return pd.DataFrame(columns=["ticker", "score", "setup_type"])
    scored = combine(factors, weights=settings.score.weights)
    classified = classify_setups(scored)
    classified = classified.sort_values("score", ascending=False).reset_index(drop=True)
    classified["rank"] = classified.index + 1
    classified["as_of"] = clock
    return classified
