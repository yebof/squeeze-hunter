"""Round-13: fetch_bars returns chronological bars; cache dedup survives tz mixes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock


def _row(ts: datetime, close: float) -> dict:
    return {
        "ticker": "GME",
        "ts": ts,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000,
    }


@pytest.mark.asyncio
async def test_fetch_bars_sorts_storage_order(tmp_path: Path) -> None:
    """A non-chronological re-ingest must not make a stale bar 'today'."""
    cache = ParquetCache(root=tmp_path)
    base = datetime(2024, 6, 3, 4, tzinfo=UTC)
    newest = [_row(base + timedelta(days=i), 10.0 + i) for i in range(3, 6)]
    oldest = [_row(base + timedelta(days=i), 10.0 + i) for i in range(0, 3)]
    cache.write_partition("bars", "GME", pd.DataFrame(newest + oldest))  # oldest LAST on disk

    provider = BacktestProvider(cache=cache, clock=Clock(now=base + timedelta(days=10)))
    bars = await provider.fetch_bars("GME", base - timedelta(days=1), base + timedelta(days=10))
    assert [b.ts for b in bars] == sorted(b.ts for b in bars)
    assert bars[-1].close == 15.0


def test_append_partition_dedups_across_tz_naive_and_aware(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    aware = datetime(2024, 1, 2, 5, tzinfo=UTC)
    cache.write_partition("bars", "GME", pd.DataFrame([_row(aware, 10.0)]))
    naive = pd.DataFrame([_row(aware.replace(tzinfo=None), 11.0)])
    cache.append_partition("bars", "GME", naive, dedup_keys=["ticker", "ts"])
    out = cache.read_partition("bars", "GME")
    assert len(out) == 1
    assert float(out.iloc[0]["close"]) == 11.0
