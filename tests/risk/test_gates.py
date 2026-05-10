from datetime import UTC, datetime

import pytest

from squeeze_hunter.risk.gates import (
    GateContext,
    PortfolioState,
    TradeProposal,
    evaluate_gates,
)


def _ctx() -> GateContext:
    return GateContext(
        as_of=datetime(2024, 5, 13, tzinfo=UTC),
        kill_switch_active=False,
        adv20_dollar_volume_by_ticker={"GME": 5_000_000_000},
        days_listed_by_ticker={"GME": 365},
        halted_tickers=frozenset(),
        universe_tickers=frozenset({"GME"}),
        earnings_within_3_days={"GME": False},
        portfolio_correlations={},
    )


def _state() -> PortfolioState:
    return PortfolioState(
        equity_usd=100_000.0,
        cash_usd=100_000.0,
        gross_exposure_pct=0.0,
        positions={},
        opened_today=0,
    )


def _proposal(score: float = 9.0, setup: str = "CAR", size: float = 5_000.0) -> TradeProposal:
    return TradeProposal(
        ticker="GME",
        score=score,
        setup_type=setup,
        target_position_usd=size,
        instrument="stock",
    )


def test_score_threshold_rejects() -> None:
    res = evaluate_gates(_proposal(score=7.0), _ctx(), _state(), score_threshold=8.0)
    assert not res.accepted
    assert res.reason == "score_below_threshold"


def test_weak_setup_rejects() -> None:
    res = evaluate_gates(_proposal(setup="Weak"), _ctx(), _state(), score_threshold=8.0)
    assert not res.accepted
    assert res.reason == "weak_setup"


def test_kill_switch_rejects() -> None:
    ctx = _ctx()
    ctx.kill_switch_active = True
    res = evaluate_gates(_proposal(), ctx, _state(), score_threshold=8.0)
    assert not res.accepted
    assert res.reason == "kill_switch_active"


def test_already_held_rejects() -> None:
    state = _state()
    state.positions["GME"] = 100
    res = evaluate_gates(_proposal(), _ctx(), state, score_threshold=8.0)
    assert not res.accepted
    assert res.reason == "already_held"


def test_position_cap_rejects() -> None:
    res = evaluate_gates(
        _proposal(size=20_000.0), _ctx(), _state(), score_threshold=8.0, position_cap=0.08
    )
    assert not res.accepted
    assert res.reason == "position_cap_exceeded"


def test_full_pass() -> None:
    res = evaluate_gates(_proposal(), _ctx(), _state(), score_threshold=8.0)
    assert res.accepted
    assert res.reason is None


def test_earnings_within_3d_halves_size() -> None:
    ctx = _ctx()
    ctx.earnings_within_3_days["GME"] = True
    res = evaluate_gates(_proposal(size=5_000.0), ctx, _state(), score_threshold=8.0)
    assert res.accepted
    assert res.adjusted_size_usd == pytest.approx(2_500.0)
