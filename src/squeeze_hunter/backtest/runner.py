"""Bar-based backtest loop. Reuses signals/score/risk/broker from production code.

P1 (architecture hardening): this loop no longer owns any position logic.
Exits come from `execution.decisions.decide_exit` and entries from
`execution.decisions.propose_entries` — the same pure core the live
lifecycle daemon and the premarket entry path call. The runner only turns
daily bars into `MarkSnapshot`s, fills decisions through the simulator, and
keeps the trade log / equity curve that Gate 1 reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
from squeeze_hunter.execution.book import new_position_meta, realized_pnl
from squeeze_hunter.execution.context import build_gate_context
from squeeze_hunter.execution.decisions import (
    MarkSnapshot,
    StopParams,
    apply_exit_decision,
    decide_exit,
    propose_entries,
    setup_stats_from_trades,
)
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.risk.gates import PortfolioState
from squeeze_hunter.risk.killswitch import KillswitchState, advance_killswitch, evaluate_killswitch
from squeeze_hunter.scan import run_scan
from squeeze_hunter.telemetry import PortfolioTelemetry
from squeeze_hunter.trading_calendar import trading_sessions

log = get_logger("backtest.runner")


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
    provider = BacktestProvider(
        cache=cache,
        clock=clock,
        finra_publication_lag_bdays=settings.data.finra_publication_lag_bdays,
    )
    broker = SimulatorBroker(initial_cash=cfg.initial_cash, cost_model=StockCostModel())
    # ticker → position meta (same shape as the live lifecycle daemon's).
    book: dict[str, dict] = {}
    trade_log: list[dict] = []
    equity_series: list[tuple[datetime, float]] = []
    daily_rows: list[dict] = []
    # CDX-P1-1: entries decided by day T's scan are NOT filled on day T (that
    # would be same-day-close lookahead — the scan saw T's close+volume). They
    # are queued here and executed at day T+1's OPEN, mirroring production:
    # nightly scan after T close → premarket-verify T+1 morning → entry ≈ open.
    pending_entries: list[dict] = []
    # R7.C2: the killswitch runs in the backtest too, fed by the SAME
    # PortfolioTelemetry the runtime uses, so a strategy that would have
    # tripped live is locked out here as well and Gate 1 sees it.
    telemetry = PortfolioTelemetry()
    kill = KillswitchState()
    kill_cooldown_days = settings.risk.killswitch.cooldown_days
    ks_cfg = settings.risk.killswitch
    monthly_dd_kill = -abs(settings.risk.monthly_drawdown_kill)  # YAML is positive
    stop_params = StopParams.from_settings(settings)

    # C7 + R11 + P5: iterate NYSE sessions only, from the shared calendar.
    trading_days: list[pd.Timestamp] = trading_sessions(cfg.start, cfg.end)
    if not trading_days:
        # R8.M16: warn loudly when the date range contains zero trading days —
        # otherwise the run silently produces an empty equity_curve and Gate 1
        # downstream treats it as "successful no-trade run."
        log.warning(
            "backtest_empty_trading_days",
            start=cfg.start.isoformat(),
            end=cfg.end.isoformat(),
            note="no NYSE sessions in range; both endpoints may be on a weekend/holiday",
        )

    for cur_ts in trading_days:
        # `day_label` (00:00 UTC) stamps the trade log / equity curve.
        day_label: datetime = cur_ts.to_pydatetime().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # Round-12: the provider clock sits at the END of the UTC day. Ingested
        # bars are stamped at exchange midnight in UTC (04:00/05:00, yahoo.py);
        # with a 00:00 clock every `[cur-2d, cur]` window ended BEFORE today's
        # bar. The live nightly scan runs after the close (22:00 ET ≈ 02:00 UTC
        # next day), so anything stamped ≤ 23:59:59 UTC is information live had.
        cur: datetime = day_label.replace(hour=23, minute=59, second=59, microsecond=999999)
        clock.advance_to(cur)

        # 0) CDX-P1-1: fill yesterday's accepted proposals at TODAY's open. If
        #    the killswitch tripped overnight, drop them — premarket would have.
        if pending_entries and not kill.active:
            for entry in pending_entries:
                t = str(entry["ticker"])
                try:
                    ebars = await provider.fetch_bars(t, cur - timedelta(days=2), cur)
                except LookupError:
                    continue
                if not ebars or ebars[-1].ts.date() != cur.date():
                    # No bar TODAY (halt / delist): the entry cannot fill. Round-12:
                    # a stale prior bar in the window must not be used either.
                    continue
                fill_px = ebars[-1].open
                if fill_px <= 0:
                    continue
                eqty = max(1, int(entry["size_usd"] // fill_px))
                # Round-12: entries fill at the 09:30 print → open-window slippage.
                eorder = await broker.submit_buy(t, eqty, fill_px, cur, is_open_5min=True)
                trade_log.append(
                    {
                        "ts": day_label,
                        "ticker": t,
                        "side": "buy",
                        "qty": eqty,
                        "price": eorder.avg_fill_price,
                        "reason": "entry",
                        "score": entry["score"],
                        "setup_type": entry["setup_type"],
                    }
                )
                book[t] = new_position_meta(
                    ticker=t,
                    qty=eqty,
                    entry_price=eorder.avg_fill_price or fill_px,
                    score=entry["score"],
                    setup_type=entry["setup_type"],
                    entry_commission_per_share=eorder.commission_usd / eqty,
                )
        # Queued entries are good for exactly one next-day fill attempt.
        pending_entries = []

        # 1) Scan for today's ranked candidates (BEFORE stop evaluation so
        #    current_score is fresh — I9).
        ranked = await run_scan(cfg.tickers, provider, cur, settings)
        ranked_by_ticker = ranked.set_index("ticker")["score"].to_dict() if not ranked.empty else {}
        for ticker, meta in book.items():
            if ticker in ranked_by_ticker:
                # A ticker that dropped out of the universe keeps its prior score.
                meta["current_score"] = float(ranked_by_ticker[ticker])

        # 2) Manage open positions through the shared decision core.
        marks: dict[str, float] = {}
        for ticker in list(book):
            meta = book[ticker]
            try:
                bars = await provider.fetch_bars(ticker, cur - timedelta(days=2), cur)
            except LookupError:
                continue
            if not bars or bars[-1].ts.date() != cur.date():
                # Round-12/13: no bar TODAY — do not re-run stops or advance
                # bars_held on a stale bar; carry the last mark forward.
                if meta.get("last_mark"):
                    marks[ticker] = meta["last_mark"]
                continue
            last = bars[-1]
            marks[ticker] = last.close
            meta["last_mark"] = last.close
            meta["bars_held"] += 1
            # CDX-P1-2 / R8.Q-I7: the killswitch gap arm sees today's intraday
            # low vs entry for EVERY position processed today, exits included.
            telemetry.record_position(ticker, meta["entry_price"], last.low)
            decision = decide_exit(meta, MarkSnapshot.from_bar(last), stop_params)
            apply_exit_decision(meta, decision)
            if decision.action == "hold":
                continue
            qty = min(decision.qty, broker.position_qty(ticker))
            if qty <= 0:
                continue
            order = await broker.submit_sell(ticker, qty, decision.fill_reference, cur)
            fill_price = order.avg_fill_price or decision.fill_reference
            realized, pct_return = realized_pnl(
                meta, qty=qty, fill_price=fill_price, sell_commission=order.commission_usd
            )
            trade_log.append(
                {
                    "ts": day_label,
                    "ticker": ticker,
                    "side": "sell",
                    "qty": qty,
                    "price": fill_price,
                    "reason": decision.reason if decision.action == "exit" else "signal_decay_half",
                    "realized": realized,
                    "pct_return": pct_return,
                    "setup_type": meta["setup_type"],
                    # R7.C3: only full exits are Kelly "trades"; halves are partial.
                    "partial": decision.action == "halve",
                }
            )
            if decision.action == "exit":
                book.pop(ticker, None)
            else:
                meta["qty"] -= qty

        # 3) Propose tomorrow's entries from today's scan (R7.C2: none while
        #    the killswitch is active).
        if not ranked.empty and not kill.active:
            ctx = await build_gate_context(
                provider, cache, cfg.tickers, cur, settings, kill_switch_active=kill.active
            )
            broker.mark_to_market(marks, ts=cur)
            state = PortfolioState(
                equity_usd=broker.equity,
                cash_usd=broker.cash,
                gross_exposure_pct=broker.gross_exposure_pct(marks),
                positions={t: broker.position_qty(t) for t in broker.positions},
                opened_today=0,
            )
            for d in propose_entries(
                ranked,
                state,
                ctx,
                settings,
                score_threshold=cfg.score_threshold,
                stats_for_setup=lambda s: setup_stats_from_trades(trade_log, s),
            ):
                if d.accepted:
                    pending_entries.append(
                        {
                            "ticker": d.ticker,
                            "size_usd": d.size_usd,
                            "score": d.score,
                            "setup_type": d.setup_type,
                        }
                    )

        # 4) Mark-to-market end of day, then the killswitch on the same
        #    telemetry the runtime feeds (equity curve + today's worst gap).
        broker.mark_to_market(marks, ts=cur)
        equity_series.append((day_label, broker.equity))
        daily_rows.append({"date": cur.date(), "equity": broker.equity, "cash": broker.cash})
        telemetry.record_equity(day_label, broker.equity)
        verdict = evaluate_killswitch(
            telemetry.to_killswitch_inputs(as_of=cur),
            monthly_drawdown_max=monthly_dd_kill,
            three_day_loss_max=ks_cfg.three_day_loss_max,
            gap_through_stop_max=ks_cfg.gap_through_stop_max,
            broker_outage_max_seconds=ks_cfg.broker_outage_max_seconds,
            data_stale_max_seconds=ks_cfg.data_stale_max_seconds,
        )
        kill, transition = advance_killswitch(kill, verdict, cur, kill_cooldown_days)
        if transition == "tripped":
            log.info("backtest_killswitch_tripped", date=cur.date().isoformat(), reason=kill.reason)
        # Gap marks are per-day in the backtest: clear them for tomorrow.
        telemetry.position_marks.clear()

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
