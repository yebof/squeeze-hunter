from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock


@pytest.mark.asyncio
async def test_backtest_only_returns_rows_at_or_before_clock(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    df = pd.DataFrame(
        {
            "ticker": ["GME", "GME", "GME"],
            "ts": [
                datetime(2024, 5, 10, tzinfo=UTC),
                datetime(2024, 5, 13, tzinfo=UTC),
                datetime(2024, 5, 14, tzinfo=UTC),
            ],
            "open": [16.0, 18.0, 25.0],
            "high": [17.0, 20.0, 30.0],
            "low": [15.0, 17.5, 22.0],
            "close": [16.5, 19.5, 28.0],
            "volume": [1_000_000, 5_000_000, 30_000_000],
        }
    )
    cache.write_partition("bars", "GME", df)

    clock = Clock(now=datetime(2024, 5, 13, 23, 59, tzinfo=UTC))
    p = BacktestProvider(cache=cache, clock=clock)
    bars = await p.fetch_bars(
        "GME", datetime(2024, 5, 1, tzinfo=UTC), datetime(2024, 5, 20, tzinfo=UTC)
    )
    assert len(bars) == 2  # 5/14 row excluded
    assert max(b.ts for b in bars) == datetime(2024, 5, 13, tzinfo=UTC)


@pytest.mark.asyncio
async def test_backtest_provider_fetch_option_chain_at(tmp_path: Path) -> None:
    """C6: BacktestProvider must serve historical option chains by date for the
    f5 OI-velocity signal to function in backtest.
    """
    cache = ParquetCache(root=tmp_path)
    df = pd.DataFrame(
        [
            {
                "strike": 18.0,
                "expiry": "2024-06-21",
                "right": "C",
                "open_interest": 4000,
                "volume": 50,
                "implied_vol": 0.7,
                "bid": 1.0,
                "ask": 1.1,
                "spot": 17.5,
            },
        ]
    )
    target_date = date(2024, 5, 6)
    cache.write_partition("options", f"GME__{target_date.isoformat()}", df)

    clock = Clock(now=datetime(2024, 5, 13, tzinfo=UTC))
    p = BacktestProvider(cache=cache, clock=clock)
    chain = await p.fetch_option_chain_at("GME", datetime(2024, 5, 6, tzinfo=UTC))
    assert len(chain.quotes) == 1
    assert chain.quotes[0].open_interest == 4000


@pytest.mark.asyncio
async def test_backtest_provider_fetch_option_chain_at_missing_returns_empty(
    tmp_path: Path,
) -> None:
    cache = ParquetCache(root=tmp_path)
    clock = Clock(now=datetime(2024, 5, 13, tzinfo=UTC))
    p = BacktestProvider(cache=cache, clock=clock)
    chain = await p.fetch_option_chain_at("GME", datetime(2024, 5, 6, tzinfo=UTC))
    # Returns an empty chain (no quotes), not None — so caller can distinguish from
    # "no archive entry" → 0 raw_value (which is the correct fallback).
    assert chain.quotes == []
