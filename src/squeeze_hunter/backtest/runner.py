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
from squeeze_hunter.risk.stops import StopState, evaluate_stops
from squeeze_hunter.scan import run_scan


@dataclass
class BacktestConfig:
    tickers: list[str]
    start: datetime
    end: datetime
    initial_cash: float = 100_000.0
    score_threshold: float = 8.0  # production default per design doc; tests may override


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

    # C7: iterate over trading (business) days only — skips weekends.
    # US holidays are not filtered here (minor inaccuracy, acceptable for now).
    # Iterate as pd.Timestamp objects (list() avoids the .to_pydatetime() method
    # that the type checker cannot resolve on DatetimeIndex).
    trading_days: list[pd.Timestamp] = list(pd.bdate_range(cfg.start, cfg.end, tz="UTC"))

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
            sig = evaluate_stops(stop_state, current_price=last.close)
            if sig.action == "exit":
                qty = broker.position_qty(ticker)
                if qty > 0:
                    order = await broker.submit_sell(ticker, qty, last.close, cur)
                    # C2: compute realized P&L so Kelly observed-trades counter works
                    realized = (order.avg_fill_price - st["entry_price"]) * qty
                    trade_log.append(
                        {
                            "ts": cur,
                            "ticker": ticker,
                            "side": "sell",
                            "qty": qty,
                            "price": order.avg_fill_price,
                            "reason": sig.reason or "exit",
                            "realized": realized,
                            "setup_type": st["setup_type"],
                        }
                    )
                tickers_to_remove.append(ticker)
            elif sig.action == "halve":
                qty = broker.position_qty(ticker) // 2
                if qty > 0:
                    order = await broker.submit_sell(ticker, qty, last.close, cur)
                    # C2: compute realized P&L for the halved quantity
                    realized = (order.avg_fill_price - st["entry_price"]) * qty
                    trade_log.append(
                        {
                            "ts": cur,
                            "ticker": ticker,
                            "side": "sell",
                            "qty": qty,
                            "price": order.avg_fill_price,
                            "reason": "signal_decay_half",
                            "realized": realized,
                            "setup_type": st["setup_type"],
                        }
                    )

        for ticker in tickers_to_remove:
            open_states.pop(ticker, None)

        # 3) Propose new entries from today's scan results
        if not ranked.empty:
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
                        days_away = (erow["report_at"] - cur).days
                        if 0 <= days_away <= 3:
                            flag = True
                            break
                earnings_within_3_days_map[t] = flag

            ctx = GateContext(
                as_of=cur,
                kill_switch_active=False,
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

            for _, row in ranked.iterrows():
                if state.opened_today >= 3:
                    break
                # C3: use per-setup priors so raw Kelly is positive from trade 0
                setup = str(row["setup_type"])
                setup_params = kelly_priors_for_setup(setup)

                # Per-setup observed wins/trades from trade_log
                setup_sells = [
                    r for r in trade_log if r["side"] == "sell" and r.get("setup_type") == setup
                ]
                wins = sum(1 for r in setup_sells if r.get("realized", 0) > 0)
                trades = len(setup_sells)
                wins_pl = [r["realized"] for r in setup_sells if r.get("realized", 0) > 0]
                losses_pl = [-r["realized"] for r in setup_sells if r.get("realized", 0) < 0]
                # Use observed win/loss ratio when both sides have happened.
                # When only wins or only losses are observed, fall back to the
                # per-setup prior payoff. The Bayesian shrinkage inside
                # kelly_position_pct still blends observed win-rate with prior
                # win-rate, so an all-wins streak still raises Kelly above the
                # prior baseline — just not unboundedly. This is intentionally
                # conservative (R9 reviewed and accepted).
                avg_payoff = (
                    (sum(wins_pl) / max(len(wins_pl), 1))
                    / max(sum(losses_pl) / max(len(losses_pl), 1), 1.0)
                    if wins_pl and losses_pl
                    else setup_params.prior_payoff
                )
                kelly_pct = kelly_position_pct(
                    observed_wins=wins,
                    observed_trades=trades,
                    observed_avg_payoff=avg_payoff,
                    params=setup_params,
                )
                target_size = state.equity_usd * kelly_pct
                if target_size <= 0:
                    # Safety floor — should rarely fire now that per-setup priors are positive
                    target_size = state.equity_usd * 0.04

                p = TradeProposal(
                    ticker=row["ticker"],
                    score=float(row["score"]),
                    setup_type=setup,
                    target_position_usd=target_size,
                    instrument="stock",
                )
                gate = evaluate_gates(p, ctx, state, score_threshold=cfg.score_threshold)
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
                open_states[row["ticker"]] = {
                    "entry_price": order.avg_fill_price,
                    "peak": order.avg_fill_price,
                    "current_score": float(row["score"]),
                    "entry_score": float(row["score"]),
                    "bars_held": 0,
                    "setup_type": setup,
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
