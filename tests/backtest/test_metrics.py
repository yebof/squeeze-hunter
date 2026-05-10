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
