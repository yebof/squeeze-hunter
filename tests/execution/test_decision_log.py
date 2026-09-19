"""P8 — decision log records every candidate's gate outcome and explains it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.execution.decision_log import DecisionLog, explain
from squeeze_hunter.execution.decisions import EntryDecision
from tests.backtest.test_runner_session_alignment import _bar, _run_full, _sessions, _write


def test_log_records_and_explains() -> None:
    log = DecisionLog()
    log.record(
        datetime(2025, 4, 21, tzinfo=UTC),
        [
            EntryDecision("HTZ", 9.4, "CAR", True, None, 7_100.0),
            EntryDecision("GME", 8.2, "Weak", False, "weak_setup", 0.0),
        ],
        source="backtest",
    )
    frame = log.to_frame()
    assert list(frame["reason"]) == ["accepted", "weak_setup"]
    text = explain(frame, ticker="gme")
    assert "skip" in text
    assert "weak_setup" in text
    assert "ENTER" in explain(frame, date="2025-04-21", ticker="HTZ")
    assert explain(frame, ticker="AMC").startswith("no decisions for")


@pytest.mark.asyncio
async def test_backtest_result_carries_decisions(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    d = _sessions("2024-06-03", 4)
    _write(cache, [_bar(t, 100, 101, 99, 100) for t in d])
    result = await _run_full(cache, "2024-06-03", "2024-06-06")
    assert not result.decisions.empty
    first = result.decisions.iloc[0]
    assert first["ticker"] == "GME"
    assert bool(first["accepted"])
    assert first["source"] == "backtest"
    # The helper scan ranks GME on the first session only, so exactly one
    # decision row exists and it is the accepted entry.
    assert len(result.decisions) == 1
    assert set(result.decisions["reason"]) == {"accepted"}


def test_settings_unused_guard() -> None:
    assert Settings().score.threshold == 8.0
