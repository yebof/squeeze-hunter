"""P1 — the unified position core: one `decide_exit` for backtest and live."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.schema import Bar
from squeeze_hunter.execution.book import new_position_meta, realized_pnl
from squeeze_hunter.execution.decisions import (
    MarkSnapshot,
    SetupStats,
    StopParams,
    apply_exit_decision,
    decide_exit,
    propose_entries,
    setup_stats_from_trades,
)
from squeeze_hunter.risk.gates import GateContext, PortfolioState

_PARAMS = StopParams.from_settings(Settings())


def _meta(**over):
    meta = new_position_meta(ticker="GME", qty=100, entry_price=100.0, score=10.0, setup_type="CAR")
    meta.update(over)
    return meta


def _bar(o: float, h: float, lo: float, c: float) -> Bar:
    return Bar(
        ticker="GME",
        ts=datetime(2024, 6, 3, 4, tzinfo=UTC),
        open=o,
        high=h,
        low=lo,
        close=c,
        volume=1,
    )


def test_hard_stop_from_a_bar_fills_at_the_low() -> None:
    d = decide_exit(_meta(), MarkSnapshot.from_bar(_bar(99, 100, 65, 99)), _PARAMS)
    assert d.action == "exit"
    assert d.reason == "hard_stop"
    assert d.qty == 100
    assert d.fill_reference == 65.0
    assert d.price_triggered


def test_time_stop_from_a_bar_fills_at_the_close() -> None:
    d = decide_exit(_meta(bars_held=21), MarkSnapshot.from_bar(_bar(100, 101, 99, 100.5)), _PARAMS)
    assert d.reason == "time_stop"
    assert d.fill_reference == 100.5
    assert not d.price_triggered


def test_trailing_peak_sees_the_open_before_the_low_but_not_the_close() -> None:
    meta = _meta(peak_price=100.0)
    # Wide-range UP day: open 100, low 99, close 130 — must NOT trail out.
    d = decide_exit(meta, MarkSnapshot.from_bar(_bar(100, 131, 99, 130)), _PARAMS)
    assert d.action == "hold"
    assert d.new_peak == 130.0  # close folded in AFTER evaluation
    # Gap-up open 130 then crater to 99: the open preceded the low → exit.
    d2 = decide_exit(meta, MarkSnapshot.from_bar(_bar(130, 131, 99, 128)), _PARAMS)
    assert d2.reason == "trailing_stop"


def test_live_quote_mark_updates_peak_before_evaluation() -> None:
    meta = _meta(peak_price=100.0)
    d = decide_exit(meta, MarkSnapshot.from_quote(120.0), _PARAMS)
    assert d.action == "hold"
    assert d.new_peak == 120.0
    d2 = decide_exit(meta | {"peak_price": 125.0}, MarkSnapshot.from_quote(99.0), _PARAMS)
    assert d2.reason == "trailing_stop"
    assert d2.fill_reference == 99.0


def test_halve_is_one_shot_and_unrepresentable_halve_marks_done() -> None:
    d = decide_exit(_meta(current_score=4.0), MarkSnapshot.from_quote(100.0), _PARAMS)
    assert d.action == "halve"
    assert d.qty == 50
    assert d.halved is True
    meta = _meta(current_score=4.0)
    apply_exit_decision(meta, d)
    assert meta["halved"] is True
    assert decide_exit(meta, MarkSnapshot.from_quote(100.0), _PARAMS).action == "hold"
    one = decide_exit(_meta(qty=1, current_score=4.0), MarkSnapshot.from_quote(100.0), _PARAMS)
    assert one.action == "hold"
    assert one.halved is True


def test_realized_pnl_charges_entry_commission_per_share_once() -> None:
    meta = _meta(entry_commission_per_share=0.005)
    realized, pct = realized_pnl(meta, qty=100, fill_price=110.0, sell_commission=0.5)
    assert realized == pytest.approx((110 - 100) * 100 - 0.5 - 0.5)
    assert pct == pytest.approx(realized / 10_000)


def _ctx(**over) -> GateContext:
    base = dict(
        as_of=datetime(2024, 6, 3, tzinfo=UTC),
        kill_switch_active=False,
        adv20_dollar_volume_by_ticker={"GME": 1e9, "AMC": 1e9},
        days_listed_by_ticker={"GME": 365, "AMC": 365},
        halted_tickers=frozenset(),
        universe_tickers=frozenset({"GME", "AMC"}),
        earnings_within_3_days={},
        portfolio_correlations={},
    )
    base.update(over)
    return GateContext(**base)


def _ranked(*rows):
    return pd.DataFrame([{"ticker": t, "score": s, "setup_type": st} for t, s, st in rows])


def test_propose_entries_sizes_with_priors_and_reserves_slots() -> None:
    settings = Settings()
    state = PortfolioState(equity_usd=100_000, cash_usd=100_000, gross_exposure_pct=0.0)
    decisions = propose_entries(
        _ranked(("GME", 9.0, "CAR"), ("AMC", 9.0, "CAR"), ("GME", 9.5, "CAR")),
        state,
        _ctx(),
        settings,
        score_threshold=8.0,
        stats_for_setup=lambda s: SetupStats(wins=0, trades=0, avg_payoff=None),
    )
    accepted = [d for d in decisions if d.accepted]
    assert [d.ticker for d in accepted] == ["GME", "AMC"]
    # CAR prior (0.25, 3.5): raw Kelly ≈ 0.0357, fraction 0.2 → ~0.71% of equity.
    assert accepted[0].size_usd == pytest.approx(100_000 * 0.2 * (0.25 * 3.5 - 0.75) / 3.5)
    dup = [d for d in decisions if d.ticker == "GME" and not d.accepted]
    assert len(dup) == 1
    assert dup[0].reason == "already_held"
    # The caller's state is untouched.
    assert state.opened_today == 0
    assert state.positions == {}


def test_propose_entries_respects_killswitch_and_weak() -> None:
    settings = Settings()
    state = PortfolioState(equity_usd=100_000, cash_usd=100_000, gross_exposure_pct=0.0)
    out = propose_entries(
        _ranked(("GME", 9.0, "Weak"), ("AMC", 9.0, "CAR")),
        state,
        _ctx(kill_switch_active=True),
        settings,
        score_threshold=8.0,
        stats_for_setup=lambda s: SetupStats(0, 0, None),
    )
    assert all(not d.accepted for d in out)
    assert {d.reason for d in out} == {"weak_setup", "kill_switch_active"}


def test_setup_stats_from_trades_uses_full_exits_only() -> None:
    log = [
        {"side": "sell", "setup_type": "CAR", "realized": 50, "pct_return": 0.10, "partial": True},
        {"side": "sell", "setup_type": "CAR", "realized": 50, "pct_return": 0.20, "partial": False},
        {
            "side": "sell",
            "setup_type": "CAR",
            "realized": -20,
            "pct_return": -0.05,
            "partial": False,
        },
        {"side": "sell", "setup_type": "GME", "realized": 90, "pct_return": 0.9, "partial": False},
        {"side": "buy", "setup_type": "CAR"},
    ]
    stats = setup_stats_from_trades(log, "CAR")
    assert stats.trades == 2
    assert stats.wins == 1
    assert stats.avg_payoff == pytest.approx(0.20 / 0.05)
    assert setup_stats_from_trades(log, "Mixed") == SetupStats(0, 0, None)
