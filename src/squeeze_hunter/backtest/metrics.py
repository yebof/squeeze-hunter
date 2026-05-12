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
    start = float(equity.iloc[0])
    # R10.5: a zero or negative starting equity makes the total-return ratio
    # undefined (division by zero or sign-flipping). Bail to 0.0 — there's
    # no meaningful annualized return to report from a degenerate series.
    if start <= 0:
        return 0.0
    total = float(equity.iloc[-1]) / start - 1
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    # R10.6: clamp to -1.0 when the period loss exceeds 100% (1+total <= 0).
    # `(negative) ** fractional` returns NaN, which silently corrupts every
    # downstream metric that compares to a threshold (NaN < x is False, so
    # Gate 1 silently "passes" a blow-up). The financial convention is that
    # losing 100%+ over a period == lost everything; report -1.0.
    if 1 + total <= 0:
        return -1.0
    return (1 + total) ** (365.25 / days) - 1


def sharpe(equity: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns(equity)
    # R4.3: a 1-element series gives std(ddof=1)=NaN. NaN != 0, so the original
    # `r.std(ddof=1) == 0` guard failed → return value was NaN → gate1's
    # `< threshold` check is False for NaN → Gate 1 silently passes a
    # degenerate equity curve.
    sd = r.std(ddof=1) if len(r) >= 2 else 0.0
    if len(r) <= 1 or sd == 0.0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino(equity: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns(equity)
    downside = r[r < 0]
    # R4.6: same NaN-vs-zero gap as sharpe — fix with len + finite guard.
    sd = downside.std(ddof=1) if len(downside) >= 2 else 0.0
    if len(downside) <= 1 or sd == 0.0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


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
    """Count distinct events whose ticker had any buy within the capture window.

    The window is asymmetric: lookback is ``window_days`` before the event
    (to credit predictive entries), but the forward allowance is only **1
    calendar day** after the event start.  Entries more than 1 day after the
    event are considered late chasing and do NOT count as captured.
    """
    if trade_log.empty:
        return 0
    hits = 0
    buys = trade_log[trade_log["side"] == "buy"]
    for ticker, event_ts in events:
        match = buys[
            (buys["ticker"] == ticker)
            & (buys["ts"] >= event_ts - timedelta(days=window_days))
            & (buys["ts"] <= event_ts + timedelta(days=1))
        ]
        if not match.empty:
            hits += 1
    return hits
