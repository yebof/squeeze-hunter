import pytest

from squeeze_hunter.risk.kelly import KellyParams, kelly_position_pct


def test_kelly_per_setup_priors_car_positive() -> None:
    """CAR setup default priors must give positive raw Kelly."""
    from squeeze_hunter.risk.kelly import kelly_priors_for_setup

    p = kelly_priors_for_setup("CAR")
    pct = kelly_position_pct(observed_wins=0, observed_trades=0, observed_avg_payoff=0.0, params=p)
    assert pct > 0.0


def test_kelly_per_setup_priors_gme_positive() -> None:
    from squeeze_hunter.risk.kelly import kelly_priors_for_setup

    p = kelly_priors_for_setup("GME")
    pct = kelly_position_pct(observed_wins=0, observed_trades=0, observed_avg_payoff=0.0, params=p)
    assert pct > 0.0


def test_kelly_per_setup_priors_mixed_positive() -> None:
    from squeeze_hunter.risk.kelly import kelly_priors_for_setup

    p = kelly_priors_for_setup("Mixed")
    pct = kelly_position_pct(observed_wins=0, observed_trades=0, observed_avg_payoff=0.0, params=p)
    assert pct > 0.0


def test_kelly_priors_for_unknown_setup_falls_back_to_safe_prior() -> None:
    """Unknown setup_type returns a safe (e.g., Mixed) prior, not the negative default."""
    from squeeze_hunter.risk.kelly import kelly_priors_for_setup

    p = kelly_priors_for_setup("Weak")  # or anything unknown
    assert isinstance(p, KellyParams)
    # raw kelly with these priors must not be negative
    raw = (p.prior_win_rate * p.prior_payoff - (1 - p.prior_win_rate)) / p.prior_payoff
    assert raw >= 0.0


def test_kelly_zero_when_priors_negative_and_no_obs() -> None:
    p = KellyParams(prior_win_rate=0.20, prior_payoff=2.0, fraction=0.20, cap=0.08)
    pct = kelly_position_pct(observed_wins=0, observed_trades=0, observed_avg_payoff=0.0, params=p)
    assert pct == 0.0


def test_kelly_positive_when_observed_strong() -> None:
    p = KellyParams(prior_win_rate=0.20, prior_payoff=2.0, fraction=0.20, cap=0.08, prior_n=30)
    pct = kelly_position_pct(
        observed_wins=20, observed_trades=40, observed_avg_payoff=4.0, params=p
    )
    assert 0.0 < pct <= 0.08


def test_kelly_capped_at_position_cap() -> None:
    p = KellyParams(prior_win_rate=0.5, prior_payoff=10.0, fraction=1.0, cap=0.08, prior_n=0)
    pct = kelly_position_pct(
        observed_wins=99, observed_trades=100, observed_avg_payoff=20.0, params=p
    )
    assert pct == pytest.approx(0.08)
