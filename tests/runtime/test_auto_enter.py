"""P1 step 4 — the live entry path, behind `execution.auto_enter`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.runtime import RuntimeContext
from tests.runtime.test_session_clamp import _seed

_PREMARKET = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)  # 08:00 ET
_OPEN_WINDOW = datetime(2026, 6, 10, 13, 36, tzinfo=UTC)  # 09:36 ET
_EARLY = datetime(2026, 6, 10, 13, 32, tzinfo=UTC)  # 09:32 ET, inside the no-trade window


async def _rc(tmp_path: Path, auto_enter: bool) -> RuntimeContext:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {"f6_bollinger_breakout": 1.0, "f7_volume_spike": 1.0}
    settings.execution.auto_enter = auto_enter
    settings.data.require_fresh_for_entries = False  # P6 gate has its own tests
    rc = RuntimeContext(cache=cache, settings=settings, tickers=["GME"], mode="sim")
    await rc.setup()
    rc.last_candidates = pd.DataFrame(
        [{"ticker": "GME", "score": 99.0, "setup_type": "CAR", "rank": 1, "as_of": _PREMARKET}]
    )
    return rc


@pytest.mark.asyncio
async def test_auto_enter_off_plans_nothing(tmp_path: Path) -> None:
    rc = await _rc(tmp_path, auto_enter=False)
    await rc.premarket_verify(now=_PREMARKET)
    assert rc.planned_entries == []
    await rc.tick(now=_OPEN_WINDOW)
    assert rc.lifecycle_state.positions == {}


@pytest.mark.asyncio
async def test_auto_enter_plans_then_fills_after_the_open_window(tmp_path: Path) -> None:
    rc = await _rc(tmp_path, auto_enter=True)
    await rc.premarket_verify(now=_PREMARKET)
    assert [p.ticker for p in rc.planned_entries] == ["GME"]
    assert rc.planned_entries[0].size_usd > 0

    await rc.tick(now=_EARLY)  # 09:30-09:35: never trade the opening print
    assert rc.lifecycle_state.positions == {}

    await rc.tick(now=_OPEN_WINDOW)
    meta = rc.lifecycle_state.positions["GME"]
    assert meta["qty"] > 0
    assert meta["entry_score"] == 99.0
    assert meta["setup_type"] == "CAR"
    assert meta["bars_held"] == 0
    assert meta["peak_price"] == meta["entry_price"]
    assert "GME" in rc.telemetry.position_marks
    assert rc.planned_entries == []  # consumed exactly once


@pytest.mark.asyncio
async def test_auto_enter_is_suppressed_by_the_killswitch(tmp_path: Path) -> None:
    rc = await _rc(tmp_path, auto_enter=True)
    rc.kill_switch_active = True
    rc._kill_reason = "monthly_drawdown"
    await rc.premarket_verify(now=_PREMARKET)
    assert rc.planned_entries == []
