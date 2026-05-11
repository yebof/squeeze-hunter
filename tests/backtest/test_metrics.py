from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from squeeze_hunter.backtest.metrics import (
    captured_events,
    max_drawdown,
    sharpe,
)


def _equity(returns: list[float], start: float = 100_000.0) -> pd.Series:
    eq = [start]
    for r in returns:
        eq.append(eq[-1] * (1 + r))
    idx = pd.date_range("2024-01-01", periods=len(eq), freq="B")
    return pd.Series(eq, index=idx)


def test_sharpe_zero_for_constant_equity() -> None:
    eq = _equity([0.0] * 252)
    assert sharpe(eq) == 0.0


def test_sharpe_positive_for_drifting_up() -> None:
    np.random.seed(0)
    rets = np.random.normal(0.001, 0.01, 252).tolist()
    eq = _equity(rets)
    assert sharpe(eq) > 0


def test_max_drawdown_finds_worst_point() -> None:
    eq = pd.Series([100, 120, 110, 90, 95, 100], index=pd.date_range("2024-01-01", periods=6))
    dd = max_drawdown(eq)
    assert dd == pytest.approx((90 - 120) / 120)


def test_captured_events_counts_hits() -> None:
    trades = pd.DataFrame(
        [
            {"ts": datetime(2024, 5, 13), "ticker": "GME", "side": "buy"},
            {"ts": datetime(2025, 4, 22), "ticker": "HTZ", "side": "buy"},
            {"ts": datetime(2024, 5, 14), "ticker": "TUP", "side": "buy"},
        ]
    )
    events = [
        ("GME", datetime(2024, 5, 13)),
        ("HTZ", datetime(2025, 4, 21)),
        ("TUP", datetime(2024, 5, 13)),
        ("CAR", datetime(2026, 4, 6)),
        ("OKLO", datetime(2025, 1, 28)),
        ("CAR", datetime(2022, 8, 8)),
        ("GME", datetime(2021, 1, 25)),
        ("BBBY", datetime(2022, 8, 15)),
    ]
    hit = captured_events(trades, events, window_days=5)
    assert hit == 3


def test_captured_events_rejects_late_entries() -> None:
    """I6 regression: entries after event_ts + 1 day must NOT count as hits.

    A strategy that chases a squeeze 4 days after the peak should not pass
    the captured-event gate.
    """
    trades = pd.DataFrame(
        [
            # Entered 4 days AFTER GME's event date — this is chasing, not capturing
            {"ts": datetime(2024, 5, 17), "ticker": "GME", "side": "buy"},
        ]
    )
    events = [("GME", datetime(2024, 5, 13))]
    hit = captured_events(trades, events, window_days=5)
    assert hit == 0


def test_captured_events_accepts_same_day_entry() -> None:
    """Entry on the event date itself counts as captured."""
    trades = pd.DataFrame([{"ts": datetime(2024, 5, 13), "ticker": "GME", "side": "buy"}])
    events = [("GME", datetime(2024, 5, 13))]
    hit = captured_events(trades, events, window_days=5)
    assert hit == 1


def test_captured_events_accepts_one_day_after_entry() -> None:
    """Entry one calendar day after the event still counts (captures fast moves)."""
    trades = pd.DataFrame([{"ts": datetime(2024, 5, 14), "ticker": "GME", "side": "buy"}])
    events = [("GME", datetime(2024, 5, 13))]
    hit = captured_events(trades, events, window_days=5)
    assert hit == 1


def test_captured_events_rejects_two_days_after_entry() -> None:
    """Entry two days after the event peak is too late."""
    trades = pd.DataFrame([{"ts": datetime(2024, 5, 15), "ticker": "GME", "side": "buy"}])
    events = [("GME", datetime(2024, 5, 13))]
    hit = captured_events(trades, events, window_days=5)
    assert hit == 0


def test_captured_events_accepts_pre_event_entries_within_window() -> None:
    """Entry 4 days before event still counts (predictive entry)."""
    trades = pd.DataFrame([{"ts": datetime(2024, 5, 9), "ticker": "GME", "side": "buy"}])
    events = [("GME", datetime(2024, 5, 13))]
    hit = captured_events(trades, events, window_days=5)
    assert hit == 1


def test_sharpe_uses_sample_std_ddof_1() -> None:
    """B1: Sharpe must use sample std (ddof=1), the financial convention."""
    rng = np.random.default_rng(42)
    rets = rng.normal(0.001, 0.01, 252)
    eq = (1 + pd.Series(rets)).cumprod() * 100_000
    eq.index = pd.date_range("2024-01-01", periods=len(eq), freq="B")
    sr = sharpe(eq)
    daily_returns_calc = eq.pct_change().dropna()
    expected = float(daily_returns_calc.mean() / daily_returns_calc.std(ddof=1) * np.sqrt(252))
    assert sr == pytest.approx(expected, rel=1e-6)


def test_sharpe_returns_zero_for_two_point_curve() -> None:
    """R4.3 regression: a 2-point equity curve gives 1 daily return →
    std(ddof=1) = NaN. The original guard `r.std == 0` didn't catch NaN,
    so sharpe returned NaN and gate1 silently passed (NaN < 1.0 is False).
    The fix: explicit len-check + isfinite guard.
    """
    eq = pd.Series([100_000.0, 105_000.0], index=pd.date_range("2024-01-01", periods=2, freq="B"))
    sr = sharpe(eq)
    assert sr == 0.0
    assert not np.isnan(sr)


def test_sharpe_returns_zero_for_one_point_curve() -> None:
    """Edge case: a 1-point series has no returns."""
    eq = pd.Series([100_000.0], index=pd.date_range("2024-01-01", periods=1, freq="B"))
    sr = sharpe(eq)
    assert sr == 0.0


def test_sortino_returns_zero_for_two_point_curve_one_downside() -> None:
    """R4.6 regression: same NaN-vs-zero gap on the downside std."""
    from squeeze_hunter.backtest.metrics import sortino

    # Series with exactly one negative return
    eq = pd.Series(
        [100_000.0, 95_000.0, 96_000.0],
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )
    s = sortino(eq)
    # Only 1 downside return → std(ddof=1) is NaN → guard returns 0
    assert s == 0.0
    assert not np.isnan(s)
