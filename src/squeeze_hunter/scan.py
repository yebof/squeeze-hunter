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
    # CDX-P2-4: combine() raises ValueError when NO factor maps to a weight
    # (missing YAML / every score.weights key typo'd). This is a CONFIG error,
    # not a transient runtime condition. The prior code caught it and returned
    # an empty ranking — so a broken config looked identical to "strategy has
    # no candidates" in BOTH backtest (Gate 1 would run on a no-trade result
    # and report "no edge") and production. Let it propagate: the CLI's
    # `backtest` command has a ValueError handler that prints a clean config
    # error + exits non-zero, and the live path's nightly_scan_safe logs it
    # loudly every cycle (visible, not silent). Fail loud > silent no-trade.
    scored = combine(factors, weights=settings.score.weights)
    thresholds = settings.score.setup_thresholds
    classified = classify_setups(
        scored,
        strong=float(thresholds.get("strong", 4.0)),
        mixed_floor=float(thresholds.get("mixed_floor", 3.0)),
    )
    classified = classified.sort_values("score", ascending=False).reset_index(drop=True)
    classified["rank"] = classified.index + 1
    classified["as_of"] = clock
    return classified
