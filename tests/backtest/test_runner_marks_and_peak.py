"""Round-13: no-bar days keep the last mark; trailing peak includes today's open."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.data.cache import ParquetCache
from tests.backtest.test_runner_session_alignment import _bar, _run_full, _sessions, _write


@pytest.mark.asyncio
async def test_simulator_marks_fall_back_to_last_mark_not_entry() -> None:
    broker = SimulatorBroker(initial_cash=100_000.0, cost_model=StockCostModel())
    ts = datetime(2024, 6, 3, tzinfo=UTC)
    await broker.submit_buy("GME", 100, 10.0, ts)
    broker.mark_to_market({"GME": 1.0}, ts)  # -90%
    crashed = broker.equity
    broker.mark_to_market({}, ts)  # no mark today
    assert broker.equity == pytest.approx(crashed)


@pytest.mark.asyncio
async def test_runner_keeps_last_mark_on_a_session_without_bars(tmp_path: Path) -> None:
    """Tue entry at 100, Wed close 120, Thu missing (halt): Thursday's equity
    must equal Wednesday's, not snap back to cost."""
    cache = ParquetCache(root=tmp_path)
    d = _sessions("2024-06-03", 5)  # Mon..Fri
    _write(
        cache,
        [
            _bar(d[0], 100, 101, 99, 100),
            _bar(d[1], 100, 101, 99, 100),
            _bar(d[2], 120, 121, 119, 120),
            # d[3] Thursday deliberately missing
            _bar(d[4], 120, 121, 119, 120),
        ],
    )
    result = await _run_full(cache, "2024-06-03", "2024-06-07")
    daily = result.daily_metrics.set_index("date")["equity"]
    assert daily.loc[datetime(2024, 6, 6).date()] == pytest.approx(
        daily.loc[datetime(2024, 6, 5).date()]
    )
    assert daily.loc[datetime(2024, 6, 5).date()] > 100_000


@pytest.mark.asyncio
async def test_runner_trailing_stop_sees_a_gap_up_open_before_the_low(tmp_path: Path) -> None:
    """Prior peak 100; today opens 130 and craters to 99: the open provably
    preceded the low, so the trailing stop (CAR 20%) must fire today."""
    cache = ParquetCache(root=tmp_path)
    d = _sessions("2024-06-03", 5)
    _write(
        cache,
        [
            _bar(d[0], 100, 101, 99, 100),
            _bar(d[1], 100, 101, 99, 100),  # entry at 100
            _bar(d[2], 100, 101, 99, 100),
            _bar(d[3], 130, 131, 99, 128),  # gap up, then -24% from the open
            _bar(d[4], 128, 129, 127, 128),
        ],
    )
    result = await _run_full(cache, "2024-06-03", "2024-06-07")
    sells = result.trade_log[result.trade_log["side"] == "sell"]
    trailing = sells[sells["reason"] == "trailing_stop"]
    assert len(trailing) == 1, sells.to_dict("records")
    assert trailing.iloc[0]["ts"].date() == d[3].date()
