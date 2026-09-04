"""Round-12: the Gate 1 'captured-the-event' case set is wired end to end.

The 8 events span 2021-2026 while the holdout is one year, so a per-holdout
count can never reach 5/8. The count is taken over the union of every
out-of-sample window (test windows + holdout); the in-sample train window
does not count.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from squeeze_hunter.backtest.runner import BacktestConfig, BacktestResult
from squeeze_hunter.backtest.walk_forward import WalkForwardConfig, run_walk_forward
from squeeze_hunter.config import Settings, load_settings
from squeeze_hunter.data.cache import ParquetCache


def _result(buys: list[tuple[str, datetime]], start: datetime) -> BacktestResult:
    idx = pd.date_range(start, periods=30, freq="D", tz="UTC")
    equity = pd.Series([100_000.0 + i * 10 for i in range(30)], index=idx)
    rows = [
        {
            "ts": ts,
            "ticker": t,
            "side": "buy",
            "qty": 10,
            "price": 10.0,
            "reason": "entry",
            "score": 9.0,
            "setup_type": "CAR",
        }
        for t, ts in buys
    ]
    log = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["ts", "ticker", "side", "qty", "price", "reason"])
    )
    return BacktestResult(equity_curve=equity, trade_log=log, daily_metrics=pd.DataFrame())


@pytest.mark.asyncio
async def test_captured_events_are_counted_across_all_out_of_sample_windows(
    tmp_path: Path,
) -> None:
    event_day = datetime(2024, 4, 15, tzinfo=UTC)  # inside test window 0, not holdout
    cfg = WalkForwardConfig(
        tickers=["GME"],
        train_start=datetime(2024, 1, 1, tzinfo=UTC),
        train_end=datetime(2024, 4, 1, tzinfo=UTC),
        test_windows=[(datetime(2024, 4, 2, tzinfo=UTC), datetime(2024, 5, 1, tzinfo=UTC))],
        holdout=(datetime(2024, 5, 2, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC)),
        validation_events=[("GME", event_day), ("AMC", datetime(2024, 5, 20, tzinfo=UTC))],
    )

    async def fake_run_backtest(bt: BacktestConfig, cache: ParquetCache, settings: Settings):
        if bt.start == cfg.train_start:
            # In-sample buy on the event day must NOT count.
            return _result([("GME", event_day)], bt.start)
        if bt.start == cfg.test_windows[0][0]:
            return _result([("GME", event_day)], bt.start)
        return _result([], bt.start)

    with patch("squeeze_hunter.backtest.walk_forward.run_backtest", side_effect=fake_run_backtest):
        report = await run_walk_forward(cfg, cache=ParquetCache(root=tmp_path), settings=Settings())

    assert report["captured_events_total"] == 1
    assert report["n_validation_events"] == 2
    # Gate 1 reads the holdout summary; it must carry the union count.
    assert report["holdout"]["captured_events"] == 1


def test_example_yaml_ships_the_eight_gate1_validation_events() -> None:
    yaml_path = Path(__file__).resolve().parents[2] / "config" / "settings.example.yml"
    events = load_settings(yaml_path).backtest.validation_events
    assert len(events) == 8
    assert {e.ticker for e in events} >= {"GME", "CAR", "BBBY", "TUP", "OKLO", "HTZ"}
    assert all(e.date.year >= 2021 for e in events)
