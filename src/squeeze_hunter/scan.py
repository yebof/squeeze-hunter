"""Daily scan orchestrator: universe → factors → score → setup → ranked output."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from squeeze_hunter.config import Settings
from squeeze_hunter.data.protocol import DataProvider
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.score.classifier import classify_setups
from squeeze_hunter.score.combiner import combine
from squeeze_hunter.signals.compute import compute_all_factors

log = get_logger("scan")


async def run_scan(
    tickers: list[str],
    provider: DataProvider,
    clock: datetime,
    settings: Settings,
) -> pd.DataFrame:
    factors = await compute_all_factors(tickers, provider, clock)
    if factors.empty:
        return pd.DataFrame(columns=["ticker", "score", "setup_type"])
    # R7.I6: combine() raises ValueError when no factor in the input is mapped
    # to a weight (typo'd YAML, every factor failing to compute, etc.). Without
    # this catch the exception propagates through nightly_scan_safe into the
    # scheduler — which then keeps firing forever with the same broken config.
    # Better to log loudly once and return an empty ranking so the operator
    # sees zero candidates rather than a stale stale list.
    try:
        scored = combine(factors, weights=settings.score.weights)
    except ValueError as e:
        log.error("scan_combine_failed", err=str(e))
        return pd.DataFrame(columns=["ticker", "score", "setup_type"])
    classified = classify_setups(scored)
    classified = classified.sort_values("score", ascending=False).reset_index(drop=True)
    classified["rank"] = classified.index + 1
    classified["as_of"] = clock
    return classified
