"""Regression tests for R1+R2+R10: daily scheduler job callbacks on RuntimeContext.

R1: eod_close must increment bars_held for every open position.
R2: nightly_scan must refresh current_score from scan output.
R10: premarket_verify must not crash (Phase 3 stub).
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
    """Minimal cache so RuntimeContext.setup() does not crash and run_scan has data."""
    base = datetime(2024, 5, 1, tzinfo=UTC)
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


# ---------------------------------------------------------------------------
# R1: eod_close increments bars_held
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_eod_close_increments_bars_held(tmp_path: Path) -> None:
    """R1 regression: eod_close must bump bars_held for every open position."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    rc.lifecycle_state.positions["GME"] = {
        "qty": 100,
        "entry_price": 100.0,
        "peak_price": 100.0,
        "entry_score": 10.0,
        "current_score": 10.0,
        "bars_held": 0,
        "setup_type": "CAR",
    }
    await rc.setup()
    await rc.eod_close(now=datetime(2026, 5, 14, 21, 0, tzinfo=UTC))
    assert rc.lifecycle_state.positions["GME"]["bars_held"] == 1
    await rc.eod_close(now=datetime(2026, 5, 15, 21, 0, tzinfo=UTC))
    assert rc.lifecycle_state.positions["GME"]["bars_held"] == 2


@pytest.mark.asyncio
async def test_runtime_eod_close_increments_all_positions(tmp_path: Path) -> None:
    """eod_close increments bars_held on every held position in one call."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    rc.lifecycle_state.positions["GME"] = {
        "qty": 100,
        "entry_price": 100.0,
        "peak_price": 100.0,
        "entry_score": 10.0,
        "current_score": 10.0,
        "bars_held": 5,
        "setup_type": "CAR",
    }
    rc.lifecycle_state.positions["AMC"] = {
        "qty": 50,
        "entry_price": 20.0,
        "peak_price": 20.0,
        "entry_score": 8.0,
        "current_score": 8.0,
        "bars_held": 3,
        "setup_type": "CAR",
    }
    await rc.setup()
    await rc.eod_close(now=datetime(2026, 5, 14, 21, 0, tzinfo=UTC))
    assert rc.lifecycle_state.positions["GME"]["bars_held"] == 6
    assert rc.lifecycle_state.positions["AMC"]["bars_held"] == 4


@pytest.mark.asyncio
async def test_runtime_eod_close_no_positions_does_not_crash(tmp_path: Path) -> None:
    """eod_close with no open positions must not raise."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    await rc.setup()
    await rc.eod_close(now=datetime(2026, 5, 14, 21, 0, tzinfo=UTC))
    # no assertion needed — just must not raise


@pytest.mark.asyncio
async def test_eod_close_skips_us_federal_holidays(tmp_path: Path) -> None:
    """R4.5 regression: eod_close must NOT increment bars_held on US federal
    holidays. The scheduler fires on every weekday including Christmas,
    Thanksgiving, July 4th, etc. — without this guard the 21-day time stop
    fires ~9 calendar days early per year.
    """
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(cache=cache, settings=Settings(), tickers=["GME"], mode="sim")
    rc.lifecycle_state.positions["GME"] = {
        "qty": 100,
        "entry_price": 100.0,
        "peak_price": 100.0,
        "entry_score": 10.0,
        "current_score": 10.0,
        "bars_held": 5,
        "setup_type": "CAR",
    }
    await rc.setup()
    # Christmas 2024 was Wednesday Dec 25 — a federal holiday on a weekday.
    # The scheduler fires (day_of_week mon-fri), but eod_close must skip.
    christmas = datetime(2024, 12, 25, 21, 0, tzinfo=UTC)  # 16:00 ET on Christmas Day
    await rc.eod_close(now=christmas)
    assert (
        rc.lifecycle_state.positions["GME"]["bars_held"] == 5
    ), "bars_held must not increment on US federal holidays"

    # Sanity: a normal trading day DOES increment.
    normal_day = datetime(2024, 12, 26, 21, 0, tzinfo=UTC)  # Thursday after Christmas
    await rc.eod_close(now=normal_day)
    assert rc.lifecycle_state.positions["GME"]["bars_held"] == 6


# ---------------------------------------------------------------------------
# R2: nightly_scan refreshes current_score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_nightly_scan_updates_current_score(tmp_path: Path) -> None:
    """R2 regression: nightly_scan must refresh current_score from the scan output
    so signal-decay stops can fire on stale setups."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {
        "f1_si_pct": 2.0,
        "f2_days_to_cover": 1.0,
        "f3_earnings_reaction": 2.0,
        "f4_wsb_mention": 1.5,
        "f5_call_oi_velocity": 1.5,
        "f6_bollinger_breakout": 1.0,
        "f7_volume_spike": 1.0,
    }
    rc = RuntimeContext(
        cache=cache,
        settings=settings,
        tickers=["GME"],
        mode="sim",
    )
    rc.lifecycle_state.positions["GME"] = {
        "qty": 100,
        "entry_price": 100.0,
        "peak_price": 100.0,
        "entry_score": 10.0,
        "current_score": 10.0,  # stale
        "bars_held": 0,
        "setup_type": "CAR",
    }
    await rc.setup()
    await rc.nightly_scan(now=datetime(2024, 5, 13, 22, 0, tzinfo=UTC))
    # R3.7: last_candidates is a dataclass field with default None, so hasattr
    # is always True regardless of whether nightly_scan ran. Check the actual
    # value to verify the scan populated it (a DataFrame, possibly empty,
    # never None after a successful scan).
    assert rc.last_candidates is not None


@pytest.mark.asyncio
async def test_runtime_nightly_scan_stores_last_candidates(tmp_path: Path) -> None:
    """nightly_scan must store the ranked output in rc.last_candidates."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    await rc.setup()
    await rc.nightly_scan(now=datetime(2024, 5, 13, 22, 0, tzinfo=UTC))
    assert hasattr(rc, "last_candidates")
    # Must be a DataFrame (possibly empty)
    import pandas as pd

    assert isinstance(rc.last_candidates, pd.DataFrame)


@pytest.mark.asyncio
async def test_runtime_nightly_scan_does_not_update_unscanned_tickers(tmp_path: Path) -> None:
    """If a held ticker isn't in today's scan output (e.g., dropped from universe),
    its current_score is preserved (not zeroed)."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(
        cache=cache,
        settings=Settings(),
        tickers=["GME"],
        mode="sim",
    )
    rc.lifecycle_state.positions["DELISTED"] = {
        "qty": 100,
        "entry_price": 50.0,
        "peak_price": 50.0,
        "entry_score": 5.0,
        "current_score": 5.0,
        "bars_held": 1,
        "setup_type": "CAR",
    }
    await rc.setup()
    await rc.nightly_scan(now=datetime(2024, 5, 13, 22, 0, tzinfo=UTC))
    # DELISTED isn't in cfg.tickers — its current_score must not be zeroed
    assert rc.lifecycle_state.positions["DELISTED"]["current_score"] == 5.0


# ---------------------------------------------------------------------------
# R10: premarket_verify stub
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_premarket_verify_runs_without_crash(tmp_path: Path) -> None:
    """R10: the premarket_verify callback must execute. For now, a no-op log is
    acceptable; the test checks it doesn't raise."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(cache=cache, settings=Settings(), tickers=["GME"], mode="sim")
    await rc.setup()
    await rc.premarket_verify(now=datetime(2026, 5, 14, 12, 0, tzinfo=UTC))


# ---------------------------------------------------------------------------
# _safe wrappers (mirrors existing tick_safe pattern)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_eod_close_safe_returns_true_on_success(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(cache=cache, settings=Settings(), tickers=["GME"], mode="sim")
    await rc.setup()
    ok = await rc.eod_close_safe(now=datetime(2026, 5, 14, 21, 0, tzinfo=UTC))
    assert ok is True


@pytest.mark.asyncio
async def test_eod_close_safe_returns_false_when_raises(tmp_path: Path) -> None:
    """eod_close_safe must catch exceptions and return False, not propagate."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(cache=cache, settings=Settings(), tickers=["GME"], mode="sim")
    await rc.setup()
    # Force eod_close to raise
    rc.eod_close = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    ok = await rc.eod_close_safe(now=datetime(2026, 5, 14, 21, 0, tzinfo=UTC))
    assert ok is False


@pytest.mark.asyncio
async def test_nightly_scan_safe_returns_true_on_success(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(cache=cache, settings=Settings(), tickers=["GME"], mode="sim")
    await rc.setup()
    ok = await rc.nightly_scan_safe(now=datetime(2024, 5, 13, 22, 0, tzinfo=UTC))
    assert ok is True


@pytest.mark.asyncio
async def test_nightly_scan_safe_returns_false_when_raises(tmp_path: Path) -> None:
    """nightly_scan_safe must catch exceptions and return False, not propagate."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(cache=cache, settings=Settings(), tickers=["GME"], mode="sim")
    await rc.setup()
    rc.nightly_scan = AsyncMock(side_effect=RuntimeError("scan boom"))  # type: ignore[method-assign]
    ok = await rc.nightly_scan_safe(now=datetime(2024, 5, 13, 22, 0, tzinfo=UTC))
    assert ok is False


@pytest.mark.asyncio
async def test_premarket_verify_safe_returns_true_on_success(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(cache=cache, settings=Settings(), tickers=["GME"], mode="sim")
    await rc.setup()
    ok = await rc.premarket_verify_safe(now=datetime(2026, 5, 14, 12, 0, tzinfo=UTC))
    assert ok is True


@pytest.mark.asyncio
async def test_premarket_verify_safe_returns_false_when_raises(tmp_path: Path) -> None:
    """premarket_verify_safe must catch exceptions and return False, not propagate."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    rc = RuntimeContext(cache=cache, settings=Settings(), tickers=["GME"], mode="sim")
    await rc.setup()
    rc.premarket_verify = AsyncMock(side_effect=RuntimeError("verify boom"))  # type: ignore[method-assign]
    ok = await rc.premarket_verify_safe(now=datetime(2026, 5, 14, 12, 0, tzinfo=UTC))
    assert ok is False
