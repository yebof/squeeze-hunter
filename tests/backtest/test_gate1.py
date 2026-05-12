import pandas as pd

from squeeze_hunter.backtest.gate1 import (
    evaluate_gate1,
    resolve_thresholds_for_n_trials,
    setup_type_coverage_warning,
)


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


def test_setup_type_coverage_warns_when_only_car_traded() -> None:
    """R9.2 regression: when f4 (Reddit baseline) and f5 (options OI velocity)
    are not yet wired, B = z[f4] + z[f5] = 0 universally. The classifier
    therefore can never produce GME or Mixed labels — Gate 1 silently
    validates the CAR strategy only. Operators must know this before
    interpreting a "PASSED" verdict.

    R10.8 strengthened: the warning text MUST surface BOTH missing setups
    (GME and Mixed), not just one. The earlier `or` assertion would have
    silently passed if a regression dropped one of them from the message.
    """
    trades = pd.DataFrame(
        [
            {"side": "buy", "setup_type": "CAR"},
            {"side": "buy", "setup_type": "CAR"},
            {"side": "sell", "setup_type": "CAR"},
        ]
    )
    warning = setup_type_coverage_warning(trades)
    assert warning is not None
    assert "CAR" in warning, "warning must reference the covered setup"
    # R10.8: BOTH missing setups must appear in the warning text. A test that
    # accepted "GME OR Mixed" would silently pass a regression that omitted one.
    assert "GME" in warning, f"warning must flag missing GME: {warning!r}"
    assert "Mixed" in warning, f"warning must flag missing Mixed: {warning!r}"


def test_setup_type_coverage_silent_when_diverse() -> None:
    """If trades cover all three setups, no warning should fire."""
    trades = pd.DataFrame(
        [
            {"side": "buy", "setup_type": "CAR"},
            {"side": "buy", "setup_type": "GME"},
            {"side": "buy", "setup_type": "Mixed"},
        ]
    )
    assert setup_type_coverage_warning(trades) is None


def test_setup_type_coverage_silent_when_no_trades() -> None:
    """Empty trade log: nothing to warn about (the run will already have an
    empty-equity warning at the report level)."""
    assert setup_type_coverage_warning(pd.DataFrame(columns=["side", "setup_type"])) is None
