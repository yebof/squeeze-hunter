"""P7 — golden Gate 1 numbers over the synthetic universe, plus invariants.

Three Gate 1 metric formulas were wrong for twelve review rounds because
nothing pinned expected numbers on a fixed dataset. This test does. When a
change is intentional, regenerate with:

    UPDATE_GOLDEN=1 uv run pytest tests/backtest/test_golden.py

and commit the new `golden/expected.json` together with the change that
explains it.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from squeeze_hunter.backtest.metrics import hit_rate_and_payoff, max_drawdown, sharpe, sortino
from squeeze_hunter.backtest.runner import BacktestConfig, BacktestResult, run_backtest
from squeeze_hunter.config import Settings, load_settings
from squeeze_hunter.data.cache import ParquetCache
from tests.backtest.synthetic_universe import END, START, build_synthetic_universe

GOLDEN = Path(__file__).parent / "golden" / "expected.json"


def _settings() -> Settings:
    repo = Path(__file__).resolve().parents[2]
    s = load_settings(repo / "config" / "settings.example.yml")
    s.data.state_path = ""
    return s


_MEMO: dict[str, tuple[BacktestResult, Settings]] = {}


async def _run(tmp_path: Path) -> tuple[BacktestResult, Settings]:
    """The run is deterministic and slow; compute it once per test session."""
    if "run" in _MEMO:
        return _MEMO["run"]
    cache = ParquetCache(root=tmp_path / "parquet")
    tickers = build_synthetic_universe(cache)
    settings = _settings()
    cfg = BacktestConfig(
        tickers=tickers,
        start=START,
        end=END,
        initial_cash=100_000.0,
        score_threshold=settings.score.threshold,
    )
    _MEMO["run"] = (await run_backtest(cfg, cache=cache, settings=settings), settings)
    return _MEMO["run"]


def _summary(result: BacktestResult) -> dict[str, float | int]:
    eq = result.equity_curve
    hit, payoff = hit_rate_and_payoff(result.trade_log)
    log = result.trade_log
    buys = log[log["side"] == "buy"] if not log.empty else log
    sells = log[log["side"] == "sell"] if not log.empty else log
    return {
        "sessions": len(eq),
        "final_equity": round(float(eq.iloc[-1]), 4),
        "sharpe": round(sharpe(eq), 8),
        "sortino": round(sortino(eq), 8),
        "max_drawdown": round(max_drawdown(eq), 8),
        "hit_rate": round(hit, 8),
        "avg_payoff": round(payoff, 8) if np.isfinite(payoff) else "inf",
        "n_buys": len(buys),
        "n_sells": len(sells),
        "exit_reasons": {k: int(v) for k, v in sells["reason"].value_counts().sort_index().items()}
        if not sells.empty
        else {},
        "realized_total": round(float(sells["realized"].sum()), 4) if not sells.empty else 0.0,
    }


@pytest.mark.slow
@pytest.mark.asyncio
async def test_gate1_numbers_match_the_golden_file(tmp_path: Path) -> None:
    result, _ = await _run(tmp_path)
    actual = _summary(result)
    assert actual["n_buys"] > 0, "the synthetic universe must produce trades"
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"golden file regenerated at {GOLDEN}")
    assert GOLDEN.is_file(), "no golden file; run with UPDATE_GOLDEN=1 once and commit it"
    expected = json.loads(GOLDEN.read_text())
    for key, want in expected.items():
        got = actual[key]
        if isinstance(want, float) and isinstance(got, float):
            assert got == pytest.approx(want, rel=1e-6, abs=1e-9), key
        else:
            assert got == want, key


@pytest.mark.slow
@pytest.mark.asyncio
async def test_simulator_invariants_hold_over_the_whole_run(tmp_path: Path) -> None:
    result, settings = await _run(tmp_path)
    daily = result.daily_metrics
    assert (daily["cash"] >= 0).all(), "cash went negative"
    assert (daily["equity"] > 0).all()

    # Never net short: replay the trade log per ticker.
    held: dict[str, int] = {}
    for _, row in result.trade_log.iterrows():
        delta = int(row["qty"]) if row["side"] == "buy" else -int(row["qty"])
        held[row["ticker"]] = held.get(row["ticker"], 0) + delta
        assert held[row["ticker"]] >= 0, f"net short in {row['ticker']} at {row['ts']}"

    # Each entry respects the position cap and the daily new-position cap.
    buys = result.trade_log[result.trade_log["side"] == "buy"]
    cap = settings.risk.position_cap
    equity_by_day = daily.set_index("date")["equity"]
    for _, b in buys.iterrows():
        # Sized against the previous session's equity (the scan day).
        day = b["ts"].date() if hasattr(b["ts"], "date") else b["ts"]
        prior = equity_by_day[equity_by_day.index < day]
        ref_equity = float(prior.iloc[-1]) if len(prior) else 100_000.0
        notional = float(b["qty"]) * float(b["price"])
        assert notional <= ref_equity * cap * 1.02 + 1e-6, (b["ticker"], notional, ref_equity)
    per_day = buys.groupby(buys["ts"].dt.date).size() if not buys.empty else []
    assert all(n <= settings.risk.max_new_per_day for n in per_day)

    # Time stop: no position outlives its bar budget.
    stop = settings.stops.time_stop_days
    open_since: dict[str, datetime] = {}
    sessions = list(daily["date"])
    for _, row in result.trade_log.sort_values("ts").iterrows():
        t = row["ticker"]
        if row["side"] == "buy":
            open_since[t] = row["ts"]
        elif t in open_since:
            entered = open_since[t].date()
            exited = row["ts"].date()
            bars = sum(1 for d in sessions if entered < d <= exited)
            assert bars <= stop + 1, (t, entered, exited, bars)
            if held.get(t, 0) == 0:
                open_since.pop(t, None)
    assert datetime.now(UTC) > START  # keeps the import honest under ruff
