"""Bar-based backtest loop. Reuses signals/score/risk/broker from production code."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
from squeeze_hunter.risk.gates import GateContext, PortfolioState, TradeProposal, evaluate_gates
from squeeze_hunter.risk.kelly import KellyParams, kelly_position_pct
from squeeze_hunter.risk.stops import StopState, evaluate_stops
from squeeze_hunter.scan import run_scan


@dataclass
class BacktestConfig:
    tickers: list[str]
    start: datetime
    end: datetime
    initial_cash: float = 100_000.0
    score_threshold: float = 8.0  # production default per design doc; tests may override
    kelly: KellyParams = field(default_factory=KellyParams)


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
    ] = {}  # ticker → {entry_price, peak, entry_score, bars_held, setup_type}
    trade_log: list[dict] = []
    equity_series: list[tuple[datetime, float]] = []
    daily_rows: list[dict] = []

    cur = cfg.start
    while cur <= cfg.end:
        clock.advance_to(cur)

        # 1) Manage open positions
        marks: dict[str, float] = {}
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
                    trade_log.append(
                        {
                            "ts": cur,
                            "ticker": ticker,
                            "side": "sell",
                            "qty": qty,
                            "price": order.avg_fill_price,
                            "reason": sig.reason or "exit",
                        }
                    )
                open_states.pop(ticker, None)
            elif sig.action == "halve":
                qty = broker.position_qty(ticker) // 2
                if qty > 0:
                    order = await broker.submit_sell(ticker, qty, last.close, cur)
                    trade_log.append(
                        {
                            "ts": cur,
                            "ticker": ticker,
                            "side": "sell",
                            "qty": qty,
                            "price": order.avg_fill_price,
                            "reason": "signal_decay_half",
                        }
                    )

        # 2) Scan & propose
        ranked = await run_scan(cfg.tickers, provider, cur, settings)
        if not ranked.empty:
            ctx = GateContext(
                as_of=cur,
                kill_switch_active=False,
                adv20_dollar_volume_by_ticker={
                    t: 1e9 for t in cfg.tickers
                },  # placeholder large enough for backtest universe
                days_listed_by_ticker={t: 365 for t in cfg.tickers},
                halted_tickers=frozenset(),
                universe_tickers=frozenset(cfg.tickers),
                earnings_within_3_days={t: False for t in cfg.tickers},
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

            wins_so_far = sum(
                1 for r in trade_log if r["side"] == "sell" and r.get("realized", 0) > 0
            )
            trades_so_far = sum(1 for r in trade_log if r["side"] == "sell")
            avg_payoff = 2.5  # bootstrap placeholder until we track per-trade payoff explicitly
            kelly_pct = kelly_position_pct(
                observed_wins=wins_so_far,
                observed_trades=trades_so_far,
                observed_avg_payoff=avg_payoff,
                params=cfg.kelly,
            )

            for _, row in ranked.iterrows():
                if state.opened_today >= 3:
                    break
                target_size = state.equity_usd * kelly_pct
                if target_size <= 0:
                    target_size = state.equity_usd * 0.04  # fallback floor while priors warm
                p = TradeProposal(
                    ticker=row["ticker"],
                    score=float(row["score"]),
                    setup_type=str(row["setup_type"]),
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
                        "setup_type": row["setup_type"],
                    }
                )
                open_states[row["ticker"]] = {
                    "entry_price": order.avg_fill_price,
                    "peak": order.avg_fill_price,
                    "current_score": float(row["score"]),
                    "entry_score": float(row["score"]),
                    "bars_held": 0,
                    "setup_type": str(row["setup_type"]),
                }
                state.positions[row["ticker"]] = qty
                state.opened_today += 1

        # 3) Mark-to-market end of day
        broker.mark_to_market(marks, ts=cur)
        equity_series.append((cur, broker.equity))
        daily_rows.append({"date": cur.date(), "equity": broker.equity, "cash": broker.cash})
        cur = cur + timedelta(days=1)

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
