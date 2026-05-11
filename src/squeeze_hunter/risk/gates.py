"""14 pre-trade gates from the design (Section 6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TradeProposal:
    ticker: str
    score: float
    setup_type: str
    target_position_usd: float
    instrument: str  # stock, call, put


@dataclass
class PortfolioState:
    equity_usd: float
    cash_usd: float
    gross_exposure_pct: float
    positions: dict[str, int] = field(default_factory=dict)  # ticker -> qty
    opened_today: int = 0


@dataclass
class GateContext:
    as_of: datetime
    kill_switch_active: bool
    adv20_dollar_volume_by_ticker: dict[str, float]
    days_listed_by_ticker: dict[str, int]
    halted_tickers: frozenset[str]
    universe_tickers: frozenset[str]
    earnings_within_3_days: dict[str, bool]
    portfolio_correlations: dict[str, float]  # max corr with existing portfolio for each candidate


@dataclass
class GateResult:
    accepted: bool
    reason: str | None = None
    adjusted_size_usd: float | None = None


def evaluate_gates(
    p: TradeProposal,
    ctx: GateContext,
    state: PortfolioState,
    *,
    score_threshold: float = 8.0,
    max_new_per_day: int = 3,
    max_positions: int = 6,
    position_cap: float = 0.08,
    max_gross_exposure: float = 0.90,
    min_adv20_multiple: float = 100.0,
    min_days_listed: int = 30,
    max_correlation: float = 0.70,
) -> GateResult:
    # R8.S-I4: reject when equity is non-positive. The later position-cap math
    # (`size / state.equity_usd`) would raise ZeroDivisionError otherwise, and
    # tick_safe would swallow it forever — preventing the killswitch from ever
    # evaluating on a bankrupt account.
    if state.equity_usd <= 0:
        return GateResult(False, "equity_nonpositive")
    if p.score < score_threshold:
        return GateResult(False, "score_below_threshold")
    if p.setup_type == "Weak":
        return GateResult(False, "weak_setup")
    if ctx.kill_switch_active:
        return GateResult(False, "kill_switch_active")
    if state.opened_today >= max_new_per_day:
        return GateResult(False, "daily_new_position_cap")
    if len(state.positions) >= max_positions:
        return GateResult(False, "max_positions_exceeded")
    if p.ticker in state.positions:
        return GateResult(False, "already_held")

    size = p.target_position_usd
    if ctx.earnings_within_3_days.get(p.ticker, False):
        size = size * 0.5

    if size / state.equity_usd > position_cap:
        return GateResult(False, "position_cap_exceeded")
    if (state.gross_exposure_pct + size / state.equity_usd) > max_gross_exposure:
        return GateResult(False, "gross_exposure_exceeded")
    adv = ctx.adv20_dollar_volume_by_ticker.get(p.ticker, 0.0)
    if size > 0 and adv < min_adv20_multiple * size:
        return GateResult(False, "insufficient_liquidity")
    if p.ticker in ctx.halted_tickers:
        return GateResult(False, "halted")
    if ctx.days_listed_by_ticker.get(p.ticker, 0) < min_days_listed:
        return GateResult(False, "listed_too_recently")
    if p.ticker not in ctx.universe_tickers:
        return GateResult(False, "outside_universe")
    if ctx.portfolio_correlations.get(p.ticker, 0.0) > max_correlation:
        return GateResult(False, "correlation_too_high")

    return GateResult(True, None, adjusted_size_usd=size)
