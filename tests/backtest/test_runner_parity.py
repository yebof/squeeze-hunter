"""P1 acceptance: the backtest runner's exits are exactly what the shared
`decide_exit` core produces over the same bars — the runner has no private
stop logic left."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.schema import Bar
from squeeze_hunter.execution.book import new_position_meta
from squeeze_hunter.execution.decisions import (
    MarkSnapshot,
    StopParams,
    apply_exit_decision,
    decide_exit,
)
from tests.backtest.test_runner_session_alignment import _bar, _run_full, _sessions, _write


@pytest.mark.asyncio
async def test_runner_exits_match_the_shared_decision_core(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    d = _sessions("2024-06-03", 9)
    rows = [
        _bar(d[0], 100, 101, 99, 100),
        _bar(d[1], 100, 101, 99, 100),  # entry at open 100
        _bar(d[2], 102, 110, 101, 108),
        _bar(d[3], 108, 125, 107, 124),
        _bar(d[4], 124, 126, 123, 125),
        _bar(d[5], 125, 126, 98, 99),  # -21% from the 125 peak → trailing stop
        _bar(d[6], 99, 100, 98, 99),
        _bar(d[7], 99, 100, 98, 99),
        _bar(d[8], 99, 100, 98, 99),
    ]
    _write(cache, rows)
    result = await _run_full(cache, "2024-06-03", "2024-06-13")
    sells = result.trade_log[result.trade_log["side"] == "sell"]
    assert len(sells) == 1
    runner_exit = sells.iloc[0]

    # Replay the same bars through the shared core by hand.
    settings = Settings()
    settings.score.weights = {"f1_si_pct": 1.0}
    params = StopParams.from_settings(settings)
    meta = new_position_meta(ticker="GME", qty=1, entry_price=100.0, score=99.0, setup_type="CAR")
    expected = None
    for r in rows[2:]:
        bar = Bar(
            ticker="GME",
            ts=r["ts"],
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            volume=r["volume"],
        )
        meta["bars_held"] += 1
        decision = decide_exit(meta, MarkSnapshot.from_bar(bar), params)
        apply_exit_decision(meta, decision)
        if decision.action == "exit":
            expected = (bar.ts.date(), decision.reason, decision.fill_reference)
            break
    assert expected is not None
    assert runner_exit["ts"].date() == expected[0]
    assert runner_exit["reason"] == expected[1]
    # The simulator applies sell slippage to the reference; the reference
    # itself is the day's low for a price-triggered stop.
    assert expected[2] == 98.0
    assert runner_exit["price"] < 98.0
    assert runner_exit["ts"].date() == datetime(2024, 6, 10, tzinfo=UTC).date()
