"""Round-13 regressions for the Gate 1 metrics."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from squeeze_hunter.backtest.deflated_sharpe import deflated_sharpe
from squeeze_hunter.backtest.metrics import captured_events, hit_rate_and_payoff, sortino


def _curve(returns: np.ndarray) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=len(returns))
    return pd.Series(100_000 * np.cumprod(1 + returns), index=idx)


def test_sortino_uses_textbook_downside_deviation() -> None:
    """Equal-sized losses (one fixed hard stop) made the old std-of-negative-
    returns collapse to ~0 and the ratio explode to 1e13."""
    rng = np.random.default_rng(0)
    r = np.zeros(252)
    r[rng.choice(252, 8, replace=False)] = -0.006
    r[rng.choice(252, 9, replace=False)] = 0.02
    eq = _curve(r)
    rr = eq.pct_change().dropna().to_numpy()
    textbook = rr.mean() / np.sqrt(np.mean(np.minimum(rr, 0) ** 2)) * np.sqrt(252)
    assert sortino(eq) == pytest.approx(textbook, rel=1e-6)
    assert sortino(eq) < 100


def test_sortino_is_zero_without_downside() -> None:
    assert sortino(_curve(np.full(50, 0.001))) == 0.0


def test_payoff_is_a_return_ratio_not_dollar_weighted() -> None:
    """+20% on a $2k lot and -10% on a $20k lot is a 2.0 payoff, not 0.2."""
    log = pd.DataFrame(
        [
            {"ts": 1, "ticker": "A", "side": "buy", "qty": 100, "price": 20.0},
            {"ts": 2, "ticker": "A", "side": "sell", "qty": 100, "price": 24.0},
            {"ts": 3, "ticker": "B", "side": "buy", "qty": 200, "price": 100.0},
            {"ts": 4, "ticker": "B", "side": "sell", "qty": 200, "price": 90.0},
        ]
    )
    hit, payoff = hit_rate_and_payoff(log)
    assert hit == 0.5
    assert payoff == pytest.approx(2.0)


def test_captured_events_allows_the_next_trading_day_fill() -> None:
    """A Friday event's scan fills at Monday's open (T+1 trading day = 3
    calendar days). That fill must count; a Tuesday fill must not."""
    event = ("OKLO", datetime(2025, 1, 3, tzinfo=UTC))  # Friday

    def _log(fill_day: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ts": datetime(2025, 1, fill_day, tzinfo=UTC),
                    "ticker": "OKLO",
                    "side": "buy",
                    "qty": 10,
                    "price": 30.0,
                }
            ]
        )

    assert captured_events(_log(6), [event]) == 1  # Monday
    assert captured_events(_log(7), [event]) == 0  # Tuesday: late chase


def test_deflated_sharpe_single_trial_benchmark_is_zero() -> None:
    """With one trial the expected max of one N(0,1) draw is 0, not -2.78:
    a zero or negative Sharpe must not score ~1.0."""
    assert deflated_sharpe(0.0, n_trials=1, n_obs=252) == pytest.approx(0.5, abs=0.02)
    assert deflated_sharpe(-1.0, n_trials=1, n_obs=252) < 0.3
    assert deflated_sharpe(2.0, n_trials=1, n_obs=252) > 0.9
