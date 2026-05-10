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


def _validate_windows(cfg: WalkForwardConfig) -> None:
    """Raise ValueError if the walk-forward windows overlap or are out of order."""
    if cfg.train_end <= cfg.train_start:
        raise ValueError(
            f"train_end ({cfg.train_end}) must be after train_start ({cfg.train_start})"
        )

    # Holdout must always start after train_end, regardless of whether test_windows
    # is populated. Empty test_windows is a valid configuration (e.g., n_trials=1
    # backtest path) but does NOT exempt the holdout/train relationship.
    holdout_start, holdout_end = cfg.holdout
    if holdout_start >= holdout_end:
        raise ValueError(
            f"holdout is internally disordered: start {holdout_start} >= end {holdout_end}"
        )
    if holdout_start < cfg.train_end:
        raise ValueError(
            f"holdout start ({holdout_start}) must be >= train_end ({cfg.train_end}); "
            "holdout overlaps training data"
        )

    if cfg.test_windows:
        first_test_start = cfg.test_windows[0][0]
        if cfg.train_end > first_test_start:
            raise ValueError(
                f"train_end ({cfg.train_end}) must be <= test_windows[0] start "
                f"({first_test_start}); windows overlap and would leak training data"
            )

        for i, (ws, we) in enumerate(cfg.test_windows):
            if ws >= we:
                raise ValueError(
                    f"test_windows[{i}] is internally disordered: start {ws} >= end {we}"
                )

        for i in range(len(cfg.test_windows) - 1):
            _, prev_end = cfg.test_windows[i]
            next_start, _ = cfg.test_windows[i + 1]
            if prev_end > next_start:
                raise ValueError(
                    f"test_windows[{i}] and test_windows[{i + 1}] overlap: "
                    f"window {i} ends {prev_end}, window {i + 1} starts {next_start}"
                )

        last_test_end = cfg.test_windows[-1][1]
        if holdout_start < last_test_end:
            raise ValueError(
                f"holdout start ({holdout_start}) must be >= last test_windows end "
                f"({last_test_end}); holdout overlaps with test data"
            )


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
    _validate_windows(cfg)

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
