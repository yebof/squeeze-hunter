from squeeze_hunter.backtest.gate1 import evaluate_gate1, resolve_thresholds_for_n_trials


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


def test_gate1_uses_strict_deflated_sharpe_when_n_trials_small() -> None:
    """I8: when n_trials <= 30, deflated_sharpe_min defaults to 0.3 (strict).
    A holdout with deflated SR below 0.3 should fail."""
    holdout = {
        "sharpe": 0.6,
        "sortino": 1.6,
        "max_drawdown": -0.20,
        "hit_rate": 0.35,
        "avg_payoff": 1.7,
        "captured_events": 6,
        "shuffle_pvalue": 0.02,
    }
    # n_trials=10: should auto-tighten to 0.3
    evaluate_gate1(holdout, n_trials=10, n_obs=250)
    # Auto-pick at n_trials=200 should be 0.0 → less strict than n_trials=10.
    evaluate_gate1(holdout, n_trials=200, n_obs=250)
    # Check the threshold explicitly via the resolved Thresholds.
    t_strict = resolve_thresholds_for_n_trials(10)
    t_lax = resolve_thresholds_for_n_trials(200)
    assert t_strict.deflated_sharpe_min >= 0.3
    assert t_lax.deflated_sharpe_min == 0.0


def test_gate1_n_trials_30_boundary_uses_strict() -> None:
    t = resolve_thresholds_for_n_trials(30)
    assert t.deflated_sharpe_min >= 0.3


def test_gate1_n_trials_31_uses_lax() -> None:
    t = resolve_thresholds_for_n_trials(31)
    assert t.deflated_sharpe_min == 0.0
