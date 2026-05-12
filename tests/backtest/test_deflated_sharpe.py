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
