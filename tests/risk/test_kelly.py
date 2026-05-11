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
    """Unknown setup_type returns the Mixed prior (positive raw Kelly)."""
    from squeeze_hunter.risk.kelly import kelly_priors_for_setup

    p = kelly_priors_for_setup("UnknownSetupName")  # truly unknown
    assert isinstance(p, KellyParams)
    # raw kelly with these priors must not be negative
    raw = (p.prior_win_rate * p.prior_payoff - (1 - p.prior_win_rate)) / p.prior_payoff
    assert raw >= 0.0


def test_kelly_priors_for_weak_returns_zero_sizing() -> None:
    """R7.M2: a 'Weak' setup must size to zero, not silently fall through to
    Mixed priors. The classifier emits Weak when factor evidence is too thin
    for confident sizing; the gate rejects Weak by name, but if it ever slips
    past (e.g., gate ordering changes), Kelly must enforce the zero floor.
    """
    from squeeze_hunter.risk.kelly import kelly_position_pct, kelly_priors_for_setup

    p = kelly_priors_for_setup("Weak")
    assert p.fraction == 0.0
    assert p.cap == 0.0
    # Even with a strong observed track record, the zero fraction means zero size.
    pct = kelly_position_pct(
        observed_wins=20, observed_trades=20, observed_avg_payoff=10.0, params=p
    )
    assert pct == 0.0


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


def test_kelly_priors_accept_yaml_overrides() -> None:
    """R8.S-I2 regression: kelly_priors_for_setup must accept fraction/cap/prior_n
    kwargs so settings.risk.* YAML overrides actually flow through. Previously
    these were hardcoded function locals — changing kelly_fraction in YAML did
    nothing.
    """
    from squeeze_hunter.risk.kelly import kelly_priors_for_setup

    p = kelly_priors_for_setup("CAR", fraction=0.10, cap=0.04, prior_n=10)
    assert p.fraction == 0.10
    assert p.cap == 0.04
    assert p.prior_n == 10
