"""Round-12 regressions: the runner's day label must line up with real bar stamps.

Ingested bars are stamped at exchange midnight converted to UTC (04:00 / 05:00
UTC — see yahoo.py). The runner used to pin each trading day at 00:00 UTC and
fetch "today's" bar from `[cur-2d, cur]`, so on every label it actually saw the
PREVIOUS session's bar, Monday labels saw nothing, and Friday sessions were
never evaluated at all. Fixtures at 00:00 UTC hid this.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from squeeze_hunter.backtest.runner import BacktestConfig, run_backtest
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache


def _sessions(start: str, n: int) -> list[datetime]:
    """Weekday sessions stamped like real ingested data (04:00 UTC)."""
    return [
        d.to_pydatetime() + timedelta(hours=4) for d in pd.bdate_range(start, periods=n, tz="UTC")
    ]


def _bar(ts: datetime, o: float, h: float, lo: float, c: float, v: int = 5_000_000) -> dict:
    return {"ticker": "GME", "ts": ts, "open": o, "high": h, "low": lo, "close": c, "volume": v}


def _write(cache: ParquetCache, bars: list[dict]) -> None:
    cache.write_partition("bars", "GME", pd.DataFrame(bars))
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


def _scan_first_day_only():
    calls = {"n": 0}

    async def _scan(tickers, provider, clock, settings):
        calls["n"] += 1
        if calls["n"] == 1:
            return pd.DataFrame(
                [{"ticker": "GME", "score": 99.0, "setup_type": "CAR", "rank": 1, "as_of": clock}]
            )
        return pd.DataFrame(columns=["ticker", "score", "setup_type"])

    return _scan


async def _run(cache: ParquetCache, start: str, end: str) -> pd.DataFrame:
    settings = Settings()
    settings.score.weights = {"f1_si_pct": 1.0}
    cfg = BacktestConfig(
        tickers=["GME"],
        start=datetime.fromisoformat(start).replace(tzinfo=UTC),
        end=datetime.fromisoformat(end).replace(tzinfo=UTC),
        score_threshold=3.0,
    )
    with patch("squeeze_hunter.backtest.runner.run_scan", side_effect=_scan_first_day_only()):
        result = await run_backtest(cfg, cache=cache, settings=settings)
    return result.trade_log


@pytest.mark.asyncio
async def test_runner_evaluates_friday_session_with_real_bar_timestamps(tmp_path: Path) -> None:
    """Mon scan → Tue entry at 100 → Fri intraday crater to 65. The hard stop
    must fire ON FRIDAY (2024-06-07), not never / not on the following week."""
    cache = ParquetCache(root=tmp_path)
    d = _sessions("2024-06-03", 6)  # Mon 3 … Fri 7, Mon 10
    _write(
        cache,
        [
            _bar(d[0], 100, 101, 99, 100),
            _bar(d[1], 100, 102, 99, 100),
            _bar(d[2], 100, 101, 99, 100),
            _bar(d[3], 100, 101, 99, 100),
            _bar(d[4], 99, 100, 65, 99, 9_000_000),  # Friday crater
            _bar(d[5], 99, 100, 98, 99),
        ],
    )
    log = await _run(cache, "2024-06-03", "2024-06-10")
    buys = log[log["side"] == "buy"]
    assert len(buys) == 1
    assert buys.iloc[0]["ts"].date() == date(2024, 6, 4)
    hard = log[(log["side"] == "sell") & (log["reason"] == "hard_stop")]
    assert len(hard) == 1, f"hard stop must fire on the Friday bar; log={log.to_dict('records')}"
    assert hard.iloc[0]["ts"].date() == date(2024, 6, 7)
    assert hard.iloc[0]["price"] < 80.0


@pytest.mark.asyncio
async def test_runner_entry_fills_at_the_open_window_price(tmp_path: Path) -> None:
    """Entries fill at the 09:30 open print, so the cost model's open-window
    slippage (+10 bps on top of the 5 bps base) must apply: 100 → 100.15."""
    cache = ParquetCache(root=tmp_path)
    d = _sessions("2024-06-03", 3)
    _write(
        cache,
        [
            _bar(d[0], 100, 101, 99, 100),
            _bar(d[1], 100, 102, 99, 100),
            _bar(d[2], 100, 101, 99, 100),
        ],
    )
    log = await _run(cache, "2024-06-03", "2024-06-05")
    buys = log[log["side"] == "buy"]
    assert len(buys) == 1
    assert buys.iloc[0]["price"] == pytest.approx(100.15, abs=1e-6)


@pytest.mark.asyncio
async def test_runner_does_not_fill_entry_on_a_missing_session(tmp_path: Path) -> None:
    """A halted / missing Tuesday session must NOT be filled using Monday's
    stale open — the bar whose close the scan already consumed."""
    cache = ParquetCache(root=tmp_path)
    d = _sessions("2024-06-03", 4)  # Mon, Tue, Wed, Thu
    _write(
        cache,
        [
            _bar(d[0], 100, 101, 99, 100),
            # Tuesday (d[1]) deliberately missing: halt.
            _bar(d[2], 120, 121, 119, 120),
            _bar(d[3], 120, 121, 119, 120),
        ],
    )
    log = await _run(cache, "2024-06-03", "2024-06-06")
    # No trades at all → the runner returns an empty frame with no columns.
    buys = log if log.empty else log[log["side"] == "buy"]
    assert buys.empty, f"entry filled on a session with no bar: {buys.to_dict('records')}"


@pytest.mark.asyncio
async def test_runner_trailing_stop_uses_peak_before_todays_close(tmp_path: Path) -> None:
    """Wide-range UP day: open 15, low 14, close 23. Folding today's close into
    the peak BEFORE evaluating today's low manufactures a -39% 'trailing stop'
    on a day the position gained 53%. The peak must lag by one bar."""
    cache = ParquetCache(root=tmp_path)
    d = _sessions("2024-06-03", 5)
    _write(
        cache,
        [
            _bar(d[0], 15, 15.2, 14.8, 15),
            _bar(d[1], 15, 15.2, 14.8, 15),  # entry at open 15
            _bar(d[2], 15, 24, 14, 23),  # +53% close, low 14 (-6.7%)
            _bar(d[3], 23, 23.5, 22.5, 23),
            _bar(d[4], 23, 23.5, 22.5, 23),
        ],
    )
    log = await _run(cache, "2024-06-03", "2024-06-07")
    sells = log[log["side"] == "sell"]
    assert sells[sells["reason"] == "trailing_stop"].empty, sells.to_dict("records")
