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


def test_gate_rejects_equity_nonpositive() -> None:
    state = _state()
    state.equity_usd = 0.0
    res = evaluate_gates(_proposal(), _ctx(), state, score_threshold=8.0)
    assert not res.accepted
    assert res.reason == "equity_nonpositive"


def test_gate_rejects_daily_new_position_cap() -> None:
    state = _state()
    state.opened_today = 3
    res = evaluate_gates(_proposal(), _ctx(), state, score_threshold=8.0, max_new_per_day=3)
    assert not res.accepted
    assert res.reason == "daily_new_position_cap"


def test_gate_rejects_max_positions_exceeded() -> None:
    state = _state()
    state.positions = {"AAA": 10, "BBB": 10, "CCC": 10, "DDD": 10, "EEE": 10, "FFF": 10}
    res = evaluate_gates(_proposal(), _ctx(), state, score_threshold=8.0, max_positions=6)
    assert not res.accepted
    assert res.reason == "max_positions_exceeded"


def test_gate_rejects_gross_exposure_exceeded() -> None:
    state = _state()
    state.gross_exposure_pct = 0.86
    # size / equity = 5_000 / 100_000 = 0.05, under position_cap (0.08), but
    # 0.86 + 0.05 = 0.91 > max_gross_exposure (0.90).
    res = evaluate_gates(
        _proposal(size=5_000.0), _ctx(), state, score_threshold=8.0, max_gross_exposure=0.90
    )
    assert not res.accepted
    assert res.reason == "gross_exposure_exceeded"


def test_gate_rejects_insufficient_liquidity() -> None:
    ctx = _ctx()
    ctx.adv20_dollar_volume_by_ticker["GME"] = 1_000.0
    res = evaluate_gates(
        _proposal(size=5_000.0), ctx, _state(), score_threshold=8.0, min_adv20_multiple=100.0
    )
    assert not res.accepted
    assert res.reason == "insufficient_liquidity"


def test_gate_rejects_halted() -> None:
    ctx = _ctx()
    ctx.halted_tickers = frozenset({"GME"})
    res = evaluate_gates(_proposal(), ctx, _state(), score_threshold=8.0)
    assert not res.accepted
    assert res.reason == "halted"


def test_gate_rejects_listed_too_recently() -> None:
    ctx = _ctx()
    ctx.days_listed_by_ticker["GME"] = 10
    res = evaluate_gates(_proposal(), ctx, _state(), score_threshold=8.0, min_days_listed=30)
    assert not res.accepted
    assert res.reason == "listed_too_recently"


def test_gate_rejects_outside_universe() -> None:
    ctx = _ctx()
    ctx.universe_tickers = frozenset()
    res = evaluate_gates(_proposal(), ctx, _state(), score_threshold=8.0)
    assert not res.accepted
    assert res.reason == "outside_universe"


def test_gate_rejects_correlation_too_high() -> None:
    ctx = _ctx()
    ctx.portfolio_correlations["GME"] = 0.85
    res = evaluate_gates(_proposal(), ctx, _state(), score_threshold=8.0, max_correlation=0.70)
    assert not res.accepted
    assert res.reason == "correlation_too_high"


def test_earnings_halving_happens_before_position_cap() -> None:
    """A proposal that would exceed position_cap at full size (15_000 / 100_000
    = 0.15 > 0.08) must pass once earnings-within-3-days halves it to 7_500
    (0.075 <= 0.08). This proves the halving in evaluate_gates runs before the
    position_cap check, not after."""
    ctx = _ctx()
    ctx.earnings_within_3_days["GME"] = True
    res = evaluate_gates(
        _proposal(size=15_000.0), ctx, _state(), score_threshold=8.0, position_cap=0.08
    )
    assert res.accepted
    assert res.adjusted_size_usd == pytest.approx(7_500.0)
