"""The unified position core: pure decisions shared by backtest and live.

P1 of the architecture-hardening plan. Before this module the backtest
runner re-implemented peak tracking, halve legs, exit fills and Kelly /
gate sizing separately from the lifecycle daemon, and every backtest-vs-live
divergence found in review rounds 12-13 lived in that duplication. Now:

- `decide_exit(meta, mark, params)` is the ONLY caller of `evaluate_stops`.
- `propose_entries(...)` is the ONLY caller of `kelly_priors_for_setup` and
  `evaluate_gates`.

Both are pure: they never touch a broker, never mutate their inputs, and
return what the caller should do. The callers (runner, lifecycle daemon,
runtime premarket path) only translate marks in and fills out.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from squeeze_hunter.config import Settings
from squeeze_hunter.data.schema import Bar
from squeeze_hunter.risk.gates import GateContext, PortfolioState, TradeProposal, evaluate_gates
from squeeze_hunter.risk.kelly import kelly_position_pct, kelly_priors_for_setup
from squeeze_hunter.risk.stops import StopState, evaluate_stops

# --------------------------------------------------------------------------- exits


@dataclass(frozen=True, slots=True)
class StopParams:
    hard_stop: float
    time_stop_bars: int
    signal_decay_halve: float
    signal_decay_exit: float
    trailing_car: float
    trailing_gme: float
    trailing_mixed: float

    @classmethod
    def from_settings(cls, settings: Settings) -> StopParams:
        s = settings.stops
        # R9.1: YAML stores trailing magnitudes as negative thresholds.
        return cls(
            hard_stop=s.hard_stop,
            time_stop_bars=s.time_stop_days,
            signal_decay_halve=s.signal_decay_halve,
            signal_decay_exit=s.signal_decay_exit,
            trailing_car=abs(s.trailing_car),
            trailing_gme=abs(s.trailing_gme),
            trailing_mixed=abs(s.trailing_mixed),
        )


@dataclass(frozen=True, slots=True)
class MarkSnapshot:
    """What the core knows about one ticker's price for one decision step.

    `peak_before` is folded into the trailing peak BEFORE the stop arms are
    evaluated at `eval_price`; it must be a price that provably preceded
    `eval_price`. `peak_after` is folded in afterwards. Price-triggered stops
    (hard / trailing) fill at `fill_on_price_stop`; time / decay stops at
    `fill_on_other_stop`.
    """

    eval_price: float
    peak_before: float
    peak_after: float
    fill_on_price_stop: float
    fill_on_other_stop: float

    @classmethod
    def from_quote(cls, price: float) -> MarkSnapshot:
        """Live intraday tick: one price is all we know."""
        return cls(price, price, price, price, price)

    @classmethod
    def from_bar(cls, bar: Bar) -> MarkSnapshot:
        """Daily bar (backtest). CDX-P1-2: the price arms are tested at the
        day's LOW and fill there; the open provably precedes the low
        (Round-13) so it may raise the peak first; the close only counts for
        the NEXT day's peak (Round-12) and is where non-price exits fill."""
        return cls(
            eval_price=bar.low,
            peak_before=bar.open,
            peak_after=bar.close,
            fill_on_price_stop=bar.low,
            fill_on_other_stop=bar.close,
        )


@dataclass(frozen=True, slots=True)
class ExitDecision:
    action: Literal["hold", "halve", "exit"]
    reason: str | None
    qty: int
    fill_reference: float
    price_triggered: bool
    new_peak: float
    halved: bool


_PRICE_TRIGGERED = frozenset({"hard_stop", "trailing_stop"})


def decide_exit(meta: Mapping[str, Any], mark: MarkSnapshot, params: StopParams) -> ExitDecision:
    """Pure: what to do with one open position given one mark."""
    peak_eval = max(float(meta["peak_price"]), mark.peak_before)
    halved = bool(meta.get("halved", False))
    sig = evaluate_stops(
        StopState(
            entry_price=float(meta["entry_price"]),
            peak_price=peak_eval,
            current_score=float(meta.get("current_score", meta["entry_score"])),
            entry_score=float(meta["entry_score"]),
            bars_held=int(meta["bars_held"]),
            setup_type=str(meta["setup_type"]),
            halved=halved,
        ),
        current_price=mark.eval_price,
        hard_stop=params.hard_stop,
        time_stop_bars=params.time_stop_bars,
        signal_decay_halve=params.signal_decay_halve,
        signal_decay_exit=params.signal_decay_exit,
        trailing_car=params.trailing_car,
        trailing_gme=params.trailing_gme,
        trailing_mixed=params.trailing_mixed,
    )
    new_peak = max(peak_eval, mark.peak_after)
    qty_held = int(meta["qty"])
    if sig.action == "hold":
        return ExitDecision("hold", None, 0, mark.eval_price, False, new_peak, halved)
    price_triggered = sig.reason in _PRICE_TRIGGERED
    fill = mark.fill_on_price_stop if price_triggered else mark.fill_on_other_stop
    if sig.action == "halve":
        qty = qty_held // 2
        # Round-12/13: the halve is a ONE-SHOT. A 1-share position cannot be
        # halved; mark it done instead of re-deciding every step forever.
        if qty <= 0:
            return ExitDecision("hold", None, 0, mark.eval_price, False, new_peak, True)
        return ExitDecision("halve", sig.reason, qty, fill, False, new_peak, True)
    return ExitDecision("exit", sig.reason, qty_held, fill, price_triggered, new_peak, halved)


def apply_exit_decision(meta: dict[str, Any], decision: ExitDecision) -> None:
    """Fold the decision's bookkeeping (peak, halved flag) into the position.
    Quantity changes are applied by the caller once a fill is confirmed."""
    meta["peak_price"] = decision.new_peak
    meta["halved"] = decision.halved


# --------------------------------------------------------------------------- entries


@dataclass(frozen=True, slots=True)
class SetupStats:
    """Observed full-exit outcomes for one setup type (R7.C3: halve legs excluded)."""

    wins: int
    trades: int
    avg_payoff: float | None  # None when only wins or only losses were observed


def setup_stats_from_trades(trades: Iterable[Mapping[str, Any]], setup_type: str) -> SetupStats:
    sells = [
        r
        for r in trades
        if r.get("side") == "sell"
        and r.get("setup_type") == setup_type
        and not r.get("partial", False)
    ]
    wins = sum(1 for r in sells if float(r.get("realized", 0.0)) > 0)
    wins_pct = [float(r["pct_return"]) for r in sells if float(r.get("pct_return", 0.0)) > 0]
    losses_pct = [-float(r["pct_return"]) for r in sells if float(r.get("pct_return", 0.0)) < 0]
    payoff: float | None = None
    if wins_pct and losses_pct:
        payoff = (sum(wins_pct) / len(wins_pct)) / max(sum(losses_pct) / len(losses_pct), 1e-6)
    return SetupStats(wins=wins, trades=len(sells), avg_payoff=payoff)


@dataclass(frozen=True, slots=True)
class EntryDecision:
    ticker: str
    score: float
    setup_type: str
    accepted: bool
    reason: str | None
    size_usd: float


def propose_entries(
    ranked: pd.DataFrame,
    state: PortfolioState,
    ctx: GateContext,
    settings: Settings,
    *,
    score_threshold: float,
    stats_for_setup: Callable[[str], SetupStats],
) -> list[EntryDecision]:
    """Pure: size each ranked candidate with per-setup Kelly and run the
    pre-trade gates, reserving slots as candidates are accepted so several
    names from one scan see the cumulative exposure (R6). The caller's
    `state` is not mutated. Every candidate yields a decision (accepted or
    the gate reason) — the decision log P8 builds on this."""
    risk = settings.risk
    priors = {k: (v.win_rate, v.payoff) for k, v in risk.kelly_priors.items()}
    work = PortfolioState(
        equity_usd=state.equity_usd,
        cash_usd=state.cash_usd,
        gross_exposure_pct=state.gross_exposure_pct,
        positions=dict(state.positions),
        opened_today=state.opened_today,
    )
    out: list[EntryDecision] = []
    if ranked.empty:
        return out
    for _, row in ranked.iterrows():
        ticker = str(row["ticker"])
        score = float(row["score"])
        setup = str(row["setup_type"])
        if work.opened_today >= risk.max_new_per_day:
            out.append(EntryDecision(ticker, score, setup, False, "daily_new_position_cap", 0.0))
            continue
        params = kelly_priors_for_setup(
            setup,
            fraction=risk.kelly_fraction,
            cap=risk.position_cap,
            prior_n=risk.bayes_prior_n,
            priors=priors,
        )
        stats = stats_for_setup(setup)
        # Only wins or only losses observed → fall back to the prior payoff;
        # the shrinkage inside kelly_position_pct still blends the win rate.
        avg_payoff = stats.avg_payoff if stats.avg_payoff is not None else params.prior_payoff
        kelly_pct = kelly_position_pct(
            observed_wins=stats.wins,
            observed_trades=stats.trades,
            observed_avg_payoff=avg_payoff,
            params=params,
        )
        target = work.equity_usd * kelly_pct
        # R7.M2: Weak sizes to zero by construction; name the real reason.
        if setup == "Weak":
            out.append(EntryDecision(ticker, score, setup, False, "weak_setup", 0.0))
            continue
        # R7.C5: no size floor — prefer zero sizing to a forced trade.
        if target <= 0:
            out.append(EntryDecision(ticker, score, setup, False, "kelly_zero", 0.0))
            continue
        gate = evaluate_gates(
            TradeProposal(
                ticker=ticker,
                score=score,
                setup_type=setup,
                target_position_usd=target,
                instrument="stock",
            ),
            ctx,
            work,
            score_threshold=score_threshold,
            max_new_per_day=risk.max_new_per_day,
            max_positions=risk.max_positions,
            position_cap=risk.position_cap,
            max_gross_exposure=risk.max_gross_exposure,
            min_days_listed=settings.universe.min_days_listed,
            min_adv20_multiple=risk.gates.min_adv20_multiple,
            max_correlation=risk.gates.max_correlation,
        )
        if not gate.accepted:
            out.append(EntryDecision(ticker, score, setup, False, gate.reason, 0.0))
            continue
        size = gate.adjusted_size_usd or target
        out.append(EntryDecision(ticker, score, setup, True, None, size))
        # Reserve the slot: qty is unknown until the fill, 0 is enough for the
        # already_held gate; exposure accumulates for the gross cap.
        work.positions[ticker] = 0
        work.opened_today += 1
        work.gross_exposure_pct += size / work.equity_usd
    return out
