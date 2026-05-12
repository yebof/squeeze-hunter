"""Bar-based backtest loop. Reuses signals/score/risk/broker from production code."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
from squeeze_hunter.risk.gates import GateContext, PortfolioState, TradeProposal, evaluate_gates
from squeeze_hunter.risk.kelly import kelly_position_pct, kelly_priors_for_setup
from squeeze_hunter.risk.killswitch import KillSwitchInputs, evaluate_killswitch
from squeeze_hunter.risk.stops import StopState, evaluate_stops
from squeeze_hunter.scan import run_scan
from squeeze_hunter.signals.earnings_reaction import _trading_days_between


@dataclass
class BacktestConfig:
    tickers: list[str]
    start: datetime
    end: datetime
    initial_cash: float = 100_000.0
    # R8.Q-I11: this default MUST equal `config.ScoreCfg.threshold` (8.0).
    # CLI passes settings.score.threshold explicitly; tests / ad-hoc callers
    # that omit it rely on this match. Keep in sync if changed.
    score_threshold: float = 8.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    daily_metrics: pd.DataFrame


async def run_backtest(
    cfg: BacktestConfig,
    cache: ParquetCache,
    settings: Settings,
) -> BacktestResult:
    clock = Clock(now=cfg.start)
    provider = BacktestProvider(cache=cache, clock=clock)
    broker = SimulatorBroker(initial_cash=cfg.initial_cash, cost_model=StockCostModel())
    open_states: dict[
        str, dict
    ] = {}  # ticker → {entry_price, peak, entry_score, current_score, bars_held, setup_type}
    trade_log: list[dict] = []
    equity_series: list[tuple[datetime, float]] = []
    daily_rows: list[dict] = []
    # R7.C2: per-day equity series for the killswitch input. We track daily
    # closes here and feed evaluate_killswitch each day so the backtest
    # discovers strategies that would have tripped the killswitch live.
    # Without this, Gate 1 silently accepts strategies whose realized
    # drawdown exceeds the production -10% kill threshold.
    kill_active = False
    kill_first_tripped_at: datetime | None = None
    kill_cooldown_days = 7  # mirrors RuntimeContext._kill_cooldown_days

    # C7: iterate over trading (business) days only — skips weekends.
    # US holidays are not filtered here (minor inaccuracy, acceptable for now).
    # Iterate as pd.Timestamp objects (list() avoids the .to_pydatetime() method
    # that the type checker cannot resolve on DatetimeIndex).
    trading_days: list[pd.Timestamp] = list(pd.bdate_range(cfg.start, cfg.end, tz="UTC"))
    if not trading_days:
        # R8.M16: warn loudly when the date range contains zero trading days —
        # otherwise the run silently produces an empty equity_curve and Gate 1
        # downstream treats it as "successful no-trade run."
        from squeeze_hunter.logging_setup import get_logger

        get_logger("backtest.runner").warning(
            "backtest_empty_trading_days",
            start=cfg.start.isoformat(),
            end=cfg.end.isoformat(),
            note="bdate_range produced zero trading days; both endpoints may be on a weekend/holiday",
        )

    # R8.S-I2: cache YAML overrides locally so they are passed into every
    # risk-evaluating call site (kelly, stops, killswitch). Prior code used
    # the function defaults, making most YAML knobs silently dead.
    kelly_fraction = settings.risk.kelly_fraction
    position_cap = settings.risk.position_cap
    max_positions = settings.risk.max_positions
    max_gross = settings.risk.max_gross_exposure
    bayes_prior_n = settings.risk.bayes_prior_n
    monthly_dd_kill = -abs(settings.risk.monthly_drawdown_kill)  # YAML is positive
    hard_stop = settings.stops.hard_stop
    time_stop_bars = settings.stops.time_stop_days
    signal_decay_halve = settings.stops.signal_decay_halve
    signal_decay_exit = settings.stops.signal_decay_exit
    # R9.1: YAML stores trailing as negative thresholds (e.g., -0.20). The
    # evaluate_stops kwargs expect positive magnitudes; convert with abs().
    trailing_car = abs(settings.stops.trailing_car)
    trailing_gme = abs(settings.stops.trailing_gme)
    trailing_mixed = abs(settings.stops.trailing_mixed)

    for cur_ts in trading_days:
        # Convert pd.Timestamp to datetime; hour/min/sec are already 0 from bdate_range.
        cur: datetime = cur_ts.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        clock.advance_to(cur)

        # 1) Scan for today's ranked candidates (must happen BEFORE stop evaluation
        #    so we can refresh current_score — I9 fix)
        ranked = await run_scan(cfg.tickers, provider, cur, settings)

        # I9: refresh current_score for open positions from today's scan results.
        # If a ticker dropped out of the universe today, keep the previous score.
        ranked_by_ticker = ranked.set_index("ticker")["score"].to_dict() if not ranked.empty else {}
        for ticker, st in open_states.items():
            if ticker in ranked_by_ticker:
                st["current_score"] = float(ranked_by_ticker[ticker])

        # 2) Manage open positions (evaluate stops with fresh current_score)
        marks: dict[str, float] = {}
        tickers_to_remove: list[str] = []
        for ticker in list(open_states):
            try:
                bars = await provider.fetch_bars(ticker, cur - timedelta(days=2), cur)
            except LookupError:
                continue
            if not bars:
                continue
            last = bars[-1]
            marks[ticker] = last.close
            st = open_states[ticker]
            st["bars_held"] += 1
            st["peak"] = max(st["peak"], last.close)
            stop_state = StopState(
                entry_price=st["entry_price"],
                peak_price=st["peak"],
                current_score=st.get("current_score", st["entry_score"]),
                entry_score=st["entry_score"],
                bars_held=st["bars_held"],
                setup_type=st["setup_type"],
            )
            sig = evaluate_stops(
                stop_state,
                current_price=last.close,
                hard_stop=hard_stop,
                time_stop_bars=time_stop_bars,
                signal_decay_halve=signal_decay_halve,
                signal_decay_exit=signal_decay_exit,
                trailing_car=trailing_car,
                trailing_gme=trailing_gme,
                trailing_mixed=trailing_mixed,
            )
            if sig.action == "exit":
                qty = broker.position_qty(ticker)
                if qty > 0:
                    order = await broker.submit_sell(ticker, qty, last.close, cur)
                    # C2: compute realized P&L so Kelly observed-trades counter works
                    # R9.10 + R10.4: subtract per-share entry commission * qty
                    # plus this leg's sell commission. The per-share form means a
                    # halve-then-exit cycle charges the entry commission exactly
                    # once across legs, not double-counted on the final exit.
                    gross = (order.avg_fill_price - st["entry_price"]) * qty
                    entry_comm_share = st.get("entry_commission_per_share", 0.0)
                    realized = gross - entry_comm_share * qty - order.commission_usd
                    # R7.C3+C4: record pct return for Kelly's b. Dollar PnL biases
                    # avg_payoff toward larger-cap trades; pct normalizes across
                    # position sizes and is invariant to equity growth.
                    cost_basis = st["entry_price"] * qty
                    pct_return = realized / cost_basis if cost_basis > 0 else 0.0
                    trade_log.append(
                        {
                            "ts": cur,
                            "ticker": ticker,
                            "side": "sell",
                            "qty": qty,
                            "price": order.avg_fill_price,
                            "reason": sig.reason or "exit",
                            "realized": realized,
                            "pct_return": pct_return,
                            "setup_type": st["setup_type"],
                            # R7.C3: full exits are the "trade" unit for Kelly.
                            "partial": False,
                        }
                    )
                tickers_to_remove.append(ticker)
            elif sig.action == "halve":
                qty = broker.position_qty(ticker) // 2
                if qty > 0:
                    order = await broker.submit_sell(ticker, qty, last.close, cur)
                    # C2: compute realized P&L for the halved quantity
                    # R9.10 + R10.4: charge proportional entry commission for THIS
                    # leg's qty, plus this leg's sell commission. The remaining
                    # qty still carries its share of the entry commission via
                    # `entry_commission_per_share`, so the final exit charges
                    # only the leftover portion.
                    gross = (order.avg_fill_price - st["entry_price"]) * qty
                    entry_comm_share = st.get("entry_commission_per_share", 0.0)
                    realized = gross - entry_comm_share * qty - order.commission_usd
                    cost_basis = st["entry_price"] * qty
                    pct_return = realized / cost_basis if cost_basis > 0 else 0.0
                    trade_log.append(
                        {
                            "ts": cur,
                            "ticker": ticker,
                            "side": "sell",
                            "qty": qty,
                            "price": order.avg_fill_price,
                            "reason": "signal_decay_half",
                            "realized": realized,
                            "pct_return": pct_return,
                            "setup_type": st["setup_type"],
                            # R7.C3: halve sells are partial — exclude from Kelly's
                            # observed-trades counter to prevent double-counting a
                            # single position as both win (halve leg) AND loss
                            # (remainder close) in the same setup bucket.
                            "partial": True,
                        }
                    )

        for ticker in tickers_to_remove:
            open_states.pop(ticker, None)

        # 3) Propose new entries from today's scan results.
        # R7.C2: skip new entries entirely when the killswitch is active.
        # This mirrors RuntimeContext.nightly_scan's suppression behavior.
        if not ranked.empty and not kill_active:
            # I10: compute earnings proximity gate for each candidate (calendar days for now).
            # We read the earnings cache directly (not through the provider's lookahead guard)
            # because knowing an earnings date 3 days in advance is public information —
            # it is NOT lookahead bias to act on a known future report date.
            earnings_within_3_days_map: dict[str, bool] = {}
            earnings_df = cache.read_partition("earnings", "all")
            if not earnings_df.empty:
                earnings_df["report_at"] = pd.to_datetime(earnings_df["report_at"], utc=True)
            for t in cfg.tickers:
                flag = False
                if not earnings_df.empty:
                    t_events = earnings_df[earnings_df["ticker"] == t]
                    for _, erow in t_events.iterrows():
                        # R7.M1: trading days, not calendar days. A Friday with
                        # earnings on Monday is 1 trading day away; the calendar-
                        # day version would have read 3 and halved size. The
                        # design's "fewer high-quality sources, simple/stable
                        # methods" doctrine prefers trading-day consistency.
                        days_away = _trading_days_between(cur, erow["report_at"])
                        if 0 <= days_away <= 3:
                            flag = True
                            break
                earnings_within_3_days_map[t] = flag

            ctx = GateContext(
                as_of=cur,
                # R7.C2: pass the live killswitch state into the gate context
                # so evaluate_gates can also reject entries when tripped.
                kill_switch_active=kill_active,
                adv20_dollar_volume_by_ticker={
                    t: 1e9 for t in cfg.tickers
                },  # placeholder large enough for backtest universe
                days_listed_by_ticker={t: 365 for t in cfg.tickers},
                halted_tickers=frozenset(),
                universe_tickers=frozenset(cfg.tickers),
                earnings_within_3_days=earnings_within_3_days_map,
                portfolio_correlations={t: 0.0 for t in cfg.tickers},
            )
            broker.mark_to_market(marks, ts=cur)
            state = PortfolioState(
                equity_usd=broker.equity,
                cash_usd=broker.cash,
                gross_exposure_pct=broker.gross_exposure_pct(marks),
                positions={t: broker.position_qty(t) for t in broker.positions},
                opened_today=0,
            )

            # R7.I5: read max_new_per_day from settings so YAML overrides
            # actually take effect (previously hardcoded literal 3).
            max_new = settings.risk.max_new_per_day
            for _, row in ranked.iterrows():
                if state.opened_today >= max_new:
                    break
                # C3: use per-setup priors so raw Kelly is positive from trade 0
                setup = str(row["setup_type"])
                setup_params = kelly_priors_for_setup(
                    setup,
                    fraction=kelly_fraction,
                    cap=position_cap,
                    prior_n=bayes_prior_n,
                )

                # R7.C3: only FULL exits (partial=False) count as "trades" for
                # the Kelly observed-rate update. Halve-sells go into trade_log
                # for P&L accounting but must not inflate the win counter.
                setup_sells = [
                    r
                    for r in trade_log
                    if r["side"] == "sell"
                    and r.get("setup_type") == setup
                    and not r.get("partial", False)
                ]
                wins = sum(1 for r in setup_sells if r.get("realized", 0) > 0)
                trades = len(setup_sells)
                # R7.C4: use pct_return, not dollar realized, for avg_payoff.
                # Kelly's `b` is the return ratio (win% / |loss%|). Dollar PnL
                # biased the ratio toward larger-cap entries and shifted Kelly
                # as equity grew — both wrong.
                wins_pct = [r["pct_return"] for r in setup_sells if r.get("pct_return", 0) > 0]
                losses_pct = [-r["pct_return"] for r in setup_sells if r.get("pct_return", 0) < 0]
                # When only wins or only losses are observed, fall back to the
                # per-setup prior payoff. The Bayesian shrinkage inside
                # kelly_position_pct still blends observed win-rate with prior
                # win-rate, so an all-wins streak still raises Kelly above the
                # prior baseline — just not unboundedly.
                if wins_pct and losses_pct:
                    avg_win_pct = sum(wins_pct) / len(wins_pct)
                    avg_loss_pct = max(sum(losses_pct) / len(losses_pct), 1e-6)
                    avg_payoff = avg_win_pct / avg_loss_pct
                else:
                    avg_payoff = setup_params.prior_payoff
                kelly_pct = kelly_position_pct(
                    observed_wins=wins,
                    observed_trades=trades,
                    observed_avg_payoff=avg_payoff,
                    params=setup_params,
                )
                target_size = state.equity_usd * kelly_pct
                # R7.C5: removed the 4% safety floor. Per-setup priors all
                # produce positive raw Kelly (CAR=2.5%, GME=4.375%, Mixed=5.45%),
                # so the floor only ever fired during losing streaks — exactly
                # when sizing should drop, not floor. Per the conservative-bias
                # design principle, prefer zero sizing over a forced trade when
                # Kelly says zero.
                if target_size <= 0:
                    continue

                p = TradeProposal(
                    ticker=row["ticker"],
                    score=float(row["score"]),
                    setup_type=setup,
                    target_position_usd=target_size,
                    instrument="stock",
                )
                gate = evaluate_gates(
                    p,
                    ctx,
                    state,
                    score_threshold=cfg.score_threshold,
                    max_new_per_day=max_new,
                    max_positions=max_positions,
                    position_cap=position_cap,
                    max_gross_exposure=max_gross,
                )
                if not gate.accepted:
                    continue
                size_usd = gate.adjusted_size_usd or target_size
                bars = await provider.fetch_bars(row["ticker"], cur - timedelta(days=2), cur)
                if not bars:
                    continue
                px = bars[-1].close
                qty = max(1, int(size_usd // px))
                order = await broker.submit_buy(row["ticker"], qty, px, cur)
                trade_log.append(
                    {
                        "ts": cur,
                        "ticker": row["ticker"],
                        "side": "buy",
                        "qty": qty,
                        "price": order.avg_fill_price,
                        "reason": "entry",
                        "score": float(row["score"]),
                        "setup_type": setup,
                    }
                )
                # R10.4: store per-share entry commission so each subsequent
                # sell (halve OR full exit) can charge a proportional share.
                # Storing the lump-sum entry commission and only deducting it
                # on the final exit double-charged the commission on
                # halve→halve→exit cycles (the lump kept getting re-charged
                # against the shrinking remainder). qty>0 was already
                # validated above (line 336: `qty = max(1, ...)`), so the
                # division is safe.
                entry_commission_per_share = order.commission_usd / qty
                open_states[row["ticker"]] = {
                    "entry_price": order.avg_fill_price,
                    "peak": order.avg_fill_price,
                    "current_score": float(row["score"]),
                    "entry_score": float(row["score"]),
                    "bars_held": 0,
                    "setup_type": setup,
                    # R9.10 + R10.4: per-share entry commission. Each sell
                    # multiplies by sold-qty, so halve→halve→exit charges the
                    # full entry commission exactly once across the legs.
                    "entry_commission_per_share": entry_commission_per_share,
                }
                state.positions[row["ticker"]] = qty
                state.opened_today += 1
                # R6: update gross_exposure_pct so the next gate evaluation in
                # this same daily loop sees the cumulative exposure. Without
                # this, three new entries can each pass the gate when they
                # would collectively breach the 90% cap.
                state.gross_exposure_pct += size_usd / state.equity_usd

        # 4) Mark-to-market end of day
        broker.mark_to_market(marks, ts=cur)
        equity_series.append((cur, broker.equity))
        daily_rows.append({"date": cur.date(), "equity": broker.equity, "cash": broker.cash})

        # R7.C2: evaluate the killswitch against the per-day equity series.
        # Strategy realism: if a strategy would have tripped live, it must
        # also be locked-out in backtest so Gate 1 sees the same behavior.
        # R8.Q-I7: compute worst per-position gap from today's intraday low
        # vs entry so the gap-through-stop killswitch arm is actually
        # exercised in backtest. Without this, a single-position blowup of
        # -25%+ silently passes Gate 1.
        worst_gap = 0.0
        for ticker, st in open_states.items():
            entry = st["entry_price"]
            if entry <= 0:
                continue
            low = marks.get(ticker)  # marks already holds today's close
            if low is None or low <= 0:
                continue
            gap = (low - entry) / entry
            if gap < worst_gap:
                worst_gap = gap
        kill_inputs = _build_killswitch_inputs(
            cur,
            equity_series,
            worst_position_gap_pct=worst_gap,
        )
        # R8.S-I2: pass YAML monthly_drawdown_kill into the killswitch so
        # the YAML override is actually wired. The other thresholds are not
        # in YAML today but keeping them as kwargs leaves room to plumb them
        # if added later.
        ks = evaluate_killswitch(kill_inputs, monthly_drawdown_max=monthly_dd_kill)
        if ks.tripped and not kill_active:
            kill_active = True
            kill_first_tripped_at = cur
        elif kill_active and kill_first_tripped_at is not None:
            cooldown_end = kill_first_tripped_at + timedelta(days=kill_cooldown_days)
            # R8.C1 mirror: don't auto-rearm sticky on persistent badness.
            # Once the cooldown expires we follow the live verdict; a new
            # sticky window opens only on a fresh clear → tripped transition.
            if cur >= cooldown_end and not ks.tripped:
                kill_active = False
                kill_first_tripped_at = None

    eq = pd.Series(
        data=[e for _, e in equity_series],
        index=pd.DatetimeIndex([t for t, _ in equity_series]),
        name="equity",
    )
    return BacktestResult(
        equity_curve=eq,
        trade_log=pd.DataFrame(trade_log),
        daily_metrics=pd.DataFrame(daily_rows),
    )


def _build_killswitch_inputs(
    as_of: datetime,
    equity_series: list[tuple[datetime, float]],
    worst_position_gap_pct: float = 0.0,
) -> KillSwitchInputs:
    """R7.C2: derive the killswitch inputs from the backtest series. The
    backtest has no broker / data-source heartbeats so those two arms are
    always zero.

    R8.Q-I7: `worst_position_gap_pct` is now a caller-supplied value (the
    backtest can compute it from intraday bars). Defaults to 0 if the caller
    doesn't supply it so the function stays usable for tests / partial setups.

    R8.Q-C3: cutoffs are made tz-aware to match the tz-aware equity series
    timestamps emitted by `pd.bdate_range(..., tz="UTC")` in the main loop.
    Mixing tz-aware and tz-naive datetimes raises TypeError on comparison.
    """
    # Normalize to the same tz as the as_of we received so comparisons work.
    tz = as_of.tzinfo
    if len(equity_series) < 2:
        return KillSwitchInputs(
            as_of=as_of,
            rolling_30d_max_drawdown=0.0,
            last_3_days_cumulative_pnl_pct=0.0,
            worst_position_gap_pct=worst_position_gap_pct,
            broker_disconnected_for_seconds=0,
            critical_data_stale_for_seconds=0,
        )
    # 30-day rolling drawdown: peak vs current within the trailing 30 days.
    cutoff_30 = as_of - timedelta(days=30)
    recent_30 = [(t, e) for t, e in equity_series if t >= cutoff_30]
    if len(recent_30) >= 2:
        peak = max(e for _, e in recent_30)
        current = recent_30[-1][1]
        dd = (current - peak) / peak if peak > 0 else 0.0
    else:
        dd = 0.0
    # 3-trading-day cumulative pnl: start equity vs current within the
    # trailing 4-business-day window.
    bdays = pd.bdate_range(end=as_of, periods=4)
    cutoff_3 = bdays[0].to_pydatetime()
    # R8.Q-C3: pd.bdate_range(end=as_of, periods=4) drops the tz of `end` —
    # the resulting cutoff is tz-naive. Restore the tz of as_of so it can be
    # compared against tz-aware equity_series timestamps.
    if tz is not None and cutoff_3.tzinfo is None:
        cutoff_3 = cutoff_3.replace(tzinfo=tz)
    recent_3 = [(t, e) for t, e in equity_series if t >= cutoff_3]
    if len(recent_3) >= 2 and recent_3[0][1] > 0:
        pnl_3d = (recent_3[-1][1] - recent_3[0][1]) / recent_3[0][1]
    else:
        pnl_3d = 0.0
    return KillSwitchInputs(
        as_of=as_of,
        rolling_30d_max_drawdown=dd,
        last_3_days_cumulative_pnl_pct=pnl_3d,
        worst_position_gap_pct=worst_position_gap_pct,
        broker_disconnected_for_seconds=0,
        critical_data_stale_for_seconds=0,
    )
