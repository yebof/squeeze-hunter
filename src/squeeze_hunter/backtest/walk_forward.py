"""Walk-forward driver: train, several test windows, holdout."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from squeeze_hunter.backtest.metrics import (
    captured_events,
    hit_rate_and_payoff,
    max_drawdown,
    sharpe,
    sortino,
)
from squeeze_hunter.backtest.runner import BacktestConfig, BacktestResult, run_backtest
from squeeze_hunter.backtest.shuffle_test import random_shuffle_pvalue
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache


@dataclass
class WalkForwardConfig:
    tickers: list[str]
    train_start: datetime
    train_end: datetime
    test_windows: list[tuple[datetime, datetime]]
    holdout: tuple[datetime, datetime]
    initial_cash: float = 100_000.0
    score_threshold: float = 8.0
    validation_events: list[tuple[str, datetime]] = field(default_factory=list)


def _summarize(result: BacktestResult, events: list[tuple[str, datetime]]) -> dict[str, Any]:
    eq = result.equity_curve
    hit, payoff = hit_rate_and_payoff(result.trade_log)
    return {
        "sharpe": sharpe(eq),
        "sortino": sortino(eq),
        "max_drawdown": max_drawdown(eq),
        "hit_rate": hit,
        "avg_payoff": payoff,
        "shuffle_pvalue": random_shuffle_pvalue(eq),
        "captured_events": captured_events(result.trade_log, events) if events else None,
        "n_trades": len(result.trade_log[result.trade_log["side"] == "buy"])
        if not result.trade_log.empty
        else 0,
    }


async def run_walk_forward(
    cfg: WalkForwardConfig,
    cache: ParquetCache,
    settings: Settings,
) -> dict[str, Any]:
    async def _bt(start: datetime, end: datetime) -> BacktestResult:
        return await run_backtest(
            BacktestConfig(
                tickers=cfg.tickers,
                start=start,
                end=end,
                initial_cash=cfg.initial_cash,
                score_threshold=cfg.score_threshold,
            ),
            cache=cache,
            settings=settings,
        )

    train_res = await _bt(cfg.train_start, cfg.train_end)
    test_results = [await _bt(s, e) for s, e in cfg.test_windows]
    holdout_res = await _bt(*cfg.holdout)
    return {
        "train": _summarize(train_res, cfg.validation_events),
        "test_windows": [_summarize(r, cfg.validation_events) for r in test_results],
        "holdout": _summarize(holdout_res, cfg.validation_events),
        "raw": {
            "train_equity": train_res.equity_curve,
            "test_equities": [r.equity_curve for r in test_results],
            "holdout_equity": holdout_res.equity_curve,
            "trades": holdout_res.trade_log,
        },
    }
