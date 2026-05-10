"""tick_safe wraps tick() and never propagates exceptions.

Regression test for review finding C8: scheduler callbacks used
asyncio.ensure_future(rc.tick(...)) directly. If tick raised, asyncio
silently logged "Task exception was never retrieved" and the scheduler
kept firing the failing callback every 60 seconds, so production errors
disappeared from the structured-log chain. tick_safe wraps tick() so
the callback can survive without dropping the error on the floor.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.runtime import RuntimeContext


def _seed(cache: ParquetCache) -> None:
    """Minimal cache so RuntimeContext.setup() does not crash."""
    base = datetime(2026, 5, 14, tzinfo=UTC)
    rows = [
        {
            "ticker": "GME",
            "ts": base + timedelta(days=i),
            "open": 18.0,
            "high": 18.5,
            "low": 17.5,
            "close": 18.0,
            "volume": 1_000_000,
        }
        for i in range(30)
    ]
    cache.write_partition("bars", "GME", pd.DataFrame(rows))
    cache.write_partition(
        "short_interest",
        "all",
        pd.DataFrame(
            columns=[
                "ticker",
                "settlement_date",
                "si_shares",
                "si_pct_float",
                "avg_daily_volume_20d",
            ]
        ),
    )
    cache.write_partition(
        "earnings",
        "all",
        pd.DataFrame(columns=["ticker", "report_at", "actual_eps", "estimate_eps"]),
    )


@pytest.mark.asyncio
async def test_tick_safe_returns_true_on_success(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    await rc.setup()
    ok = await rc.tick_safe(now=datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    assert ok is True


@pytest.mark.asyncio
async def test_tick_safe_returns_false_when_tick_raises(tmp_path: Path) -> None:
    """If tick() blows up, tick_safe must catch and return False, not propagate."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    await rc.setup()
    # Replace tick with an AsyncMock that raises
    rc.tick = AsyncMock(side_effect=RuntimeError("simulated tick failure"))  # type: ignore[method-assign]
    ok = await rc.tick_safe(now=datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    assert ok is False


@pytest.mark.asyncio
async def test_tick_safe_does_not_propagate_unhandled_exception(tmp_path: Path) -> None:
    """Even an exotic exception (e.g. AttributeError from a missing method)
    must not propagate. The scheduler relies on this contract.
    """
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    await rc.setup()
    rc.tick = AsyncMock(side_effect=AttributeError("no such method"))  # type: ignore[method-assign]
    # Should not raise
    ok = await rc.tick_safe(now=datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    assert ok is False


@pytest.mark.asyncio
async def test_tick_safe_keeps_kill_switch_state_after_exception(tmp_path: Path) -> None:
    """After tick_safe catches an exception, kill_switch_active is unchanged
    (the failure does not silently flip it). Subsequent tick_safe calls
    should still work.
    """
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    await rc.setup()
    assert rc.kill_switch_active is False

    # First tick raises
    rc.tick = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    await rc.tick_safe(now=datetime(2026, 5, 14, 14, 0, tzinfo=UTC))
    # State preserved despite the exception
    assert rc.kill_switch_active is False
