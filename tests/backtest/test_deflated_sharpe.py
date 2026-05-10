from squeeze_hunter.backtest.deflated_sharpe import deflated_sharpe


def test_deflated_sharpe_lower_with_more_trials() -> None:
    sr_no_penalty = deflated_sharpe(observed_sr=2.0, n_trials=1, n_obs=252)
    sr_with_penalty = deflated_sharpe(observed_sr=2.0, n_trials=200, n_obs=252)
    assert sr_with_penalty < sr_no_penalty
    assert sr_with_penalty > 0
