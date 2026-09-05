import pytest

from squeeze_hunter.backtest.deflated_sharpe import deflated_sharpe


def test_deflated_sharpe_lower_with_more_trials() -> None:
    sr_no_penalty = deflated_sharpe(observed_sr=2.0, n_trials=1, n_obs=252)
    sr_with_penalty = deflated_sharpe(observed_sr=2.0, n_trials=200, n_obs=252)
    assert sr_with_penalty < sr_no_penalty
    assert sr_with_penalty > 0


def test_deflated_sharpe_returns_zero_for_degenerate_n_obs() -> None:
    """R10.7 regression: when n_obs < 20 (or n_trials < 1) the formula has too
    little signal to deflate meaningfully. Prior code returned `observed_sr`
    UNDEFLATED, which silently bypasses the overfitting penalty — Gate 1
    sees the raw Sharpe and may "PASS" a degenerate short-window run.
    Returning 0.0 makes Gate 1 fail (deflated_sharpe < threshold).
    """
    # 10 observations is well below the 20-bar floor.
    assert deflated_sharpe(observed_sr=3.0, n_trials=10, n_obs=10) == 0.0
    assert deflated_sharpe(observed_sr=3.0, n_trials=0, n_obs=252) == 0.0


def test_deflated_sharpe_at_n_obs_floor() -> None:
    """At the n_obs=20 boundary the formula DOES apply (penalty starts here).
    Above the boundary, output should track observed_sr but be deflated."""
    val = deflated_sharpe(observed_sr=2.0, n_trials=10, n_obs=20)
    # The deflated value is a probability in [0, 1]; for a strong observed SR
    # it should be > 0 but well below 1.
    assert 0.0 < val < 1.0


def test_deflated_sharpe_units_realistic_annualized_sharpe() -> None:
    """Round-11 regression: observed_sr is the ANNUALIZED Sharpe (metrics.sharpe
    multiplies by sqrt(252)), but the DSR standard error and expected-max
    benchmark are in PER-PERIOD SR units. The shipped code fed the annualized SR
    straight in AND dropped the `e_max * sr_std` scaling, so Gate 1's pass/fail
    boundary became a units artifact: deflated ≈ 0 for annualized 1.0-1.5 (which
    clear sharpe_min=1.0) and jumped to ≈ 1 only near annualized 2.0. A realistic
    annualized Sharpe of 1.2 over 250 obs with 10 trials must yield a SANE
    probability, not a near-zero cliff.
    """
    val = deflated_sharpe(observed_sr=1.2, n_trials=10, n_obs=250)
    assert 0.15 < val < 0.6, f"expected a sane deflated probability, got {val}"


def test_deflated_sharpe_no_penalty_at_single_trial() -> None:
    """With n_trials=1 there is no multiple-testing penalty: the benchmark is
    0 and the value is the probability the true Sharpe is positive given the
    estimation error. Round-13: the old expectation (> 0.95 for Sharpe 1.0)
    encoded an inverted benchmark (-2.78) that also scored a NEGATIVE Sharpe
    at 0.96; an annualized 1.0 over 250 observations is ~1 standard error
    above zero, i.e. ~0.84."""
    val = deflated_sharpe(observed_sr=1.0, n_trials=1, n_obs=250)
    assert 0.75 < val < 0.95
    assert deflated_sharpe(observed_sr=0.0, n_trials=1, n_obs=250) == pytest.approx(0.5, abs=0.02)


def test_deflated_sharpe_monotonic_in_observed_sharpe() -> None:
    """Higher annualized Sharpe → higher deflated probability at fixed
    trials/obs."""
    lo = deflated_sharpe(observed_sr=1.0, n_trials=20, n_obs=250)
    hi = deflated_sharpe(observed_sr=2.5, n_trials=20, n_obs=250)
    assert hi > lo
