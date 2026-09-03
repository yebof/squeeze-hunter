"""Round-12: a Kelly proposal sized exactly AT the cap must not be rejected by
float rounding (`(equity*0.08)/equity` evaluates to 0.08000000000000002 for
~2% of equity values)."""

from __future__ import annotations

from datetime import UTC, datetime

from squeeze_hunter.risk.gates import (
    GateContext,
    PortfolioState,
    TradeProposal,
    evaluate_gates,
)


def _equity_with_rounding_artifact(cap: float = 0.08) -> float:
    equity = 1000.0
    while (equity * cap) / equity <= cap:
        equity += 0.37
    return equity


def test_gate_accepts_proposal_sized_exactly_at_position_cap() -> None:
    equity = _equity_with_rounding_artifact()
    state = PortfolioState(
        equity_usd=equity, cash_usd=equity, gross_exposure_pct=0.0, positions={}, opened_today=0
    )
    ctx = GateContext(
        as_of=datetime(2024, 5, 13, tzinfo=UTC),
        kill_switch_active=False,
        adv20_dollar_volume_by_ticker={"GME": 5_000_000_000},
        days_listed_by_ticker={"GME": 365},
        halted_tickers=frozenset(),
        universe_tickers=frozenset({"GME"}),
        earnings_within_3_days={"GME": False},
        portfolio_correlations={},
    )
    proposal = TradeProposal(
        ticker="GME",
        score=9.0,
        setup_type="CAR",
        target_position_usd=equity * 0.08,
        instrument="stock",
    )
    result = evaluate_gates(proposal, ctx, state, position_cap=0.08)
    assert result.accepted, result.reason
