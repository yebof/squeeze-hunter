from squeeze_hunter.backtest.gate1 import evaluate_gate1


def _holdout(**kw):
    base = {
        "sharpe": 1.2,
        "sortino": 1.6,
        "max_drawdown": -0.20,
        "hit_rate": 0.35,
        "avg_payoff": 1.7,
        "captured_events": 6,
        "shuffle_pvalue": 0.02,
    }
    base.update(kw)
    return base


def test_gate1_passes_clean() -> None:
    v = evaluate_gate1(holdout=_holdout(), n_trials=100, n_obs=250)
    assert v.passed
    assert v.failures == []


def test_gate1_fails_on_low_sharpe() -> None:
    v = evaluate_gate1(holdout=_holdout(sharpe=0.5), n_trials=100, n_obs=250)
    assert not v.passed
    assert "sharpe" in " ".join(v.failures)


def test_gate1_fails_on_high_drawdown() -> None:
    v = evaluate_gate1(holdout=_holdout(max_drawdown=-0.40), n_trials=100, n_obs=250)
    assert not v.passed
    assert any("drawdown" in f for f in v.failures)


def test_gate1_fails_on_too_few_captured() -> None:
    v = evaluate_gate1(holdout=_holdout(captured_events=4), n_trials=100, n_obs=250)
    assert not v.passed
    assert any("captured" in f for f in v.failures)
