"""Reporting metrics. Stateless functions over an equity curve / trade log."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0] - 1
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    return (1 + total) ** (365.25 / days) - 1


def sharpe(equity: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns(equity)
    if r.std(ddof=0) == 0 or len(r) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(periods_per_year))


def sortino(equity: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns(equity)
    downside = r[r < 0]
    if downside.std(ddof=0) == 0 or len(downside) == 0:
        return 0.0
    return float(r.mean() / downside.std(ddof=0) * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return float(dd.min())


def calmar(equity: pd.Series) -> float:
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return 0.0
    return annualized_return(equity) / dd


def hit_rate_and_payoff(trade_log: pd.DataFrame) -> tuple[float, float]:
    """Aggregate buy/sell pairs by ticker, opening lot order. Returns (hit_rate, avg_payoff)."""
    if trade_log.empty:
        return 0.0, 0.0
    pnls = []
    open_lots: dict[str, list[tuple[int, float]]] = {}
    for _, row in trade_log.iterrows():
        t = row["ticker"]
        if row["side"] == "buy":
            open_lots.setdefault(t, []).append((row["qty"], row["price"]))
        else:
            qty = row["qty"]
            price = row["price"]
            lots = open_lots.get(t, [])
            while qty > 0 and lots:
                lot_qty, lot_price = lots[0]
                use = min(qty, lot_qty)
                pnls.append((price - lot_price) * use)
                qty -= use
                if use == lot_qty:
                    lots.pop(0)
                else:
                    lots[0] = (lot_qty - use, lot_price)
    if not pnls:
        return 0.0, 0.0
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]
    hit = len(wins) / len(pnls)
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 1.0
    payoff = avg_win / avg_loss if avg_loss > 0 else 0.0
    return hit, payoff


def captured_events(
    trade_log: pd.DataFrame,
    events: list[tuple[str, datetime]],
    window_days: int = 5,
) -> int:
    """Count distinct events whose ticker had any buy within ±`window_days` of event ts."""
    if trade_log.empty:
        return 0
    hits = 0
    buys = trade_log[trade_log["side"] == "buy"]
    for ticker, event_ts in events:
        match = buys[
            (buys["ticker"] == ticker)
            & (buys["ts"] >= event_ts - timedelta(days=window_days))
            & (buys["ts"] <= event_ts + timedelta(days=window_days))
        ]
        if not match.empty:
            hits += 1
    return hits
