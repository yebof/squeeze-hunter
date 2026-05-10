"""Gate 1 verdict — does the strategy clear all conditions to start paper trading?"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from squeeze_hunter.backtest.deflated_sharpe import deflated_sharpe


@dataclass
class Gate1Thresholds:
    sharpe_min: float = 1.0
    sortino_min: float = 1.5
    max_drawdown_max: float = -0.25  # i.e. drawdown must be >= this (less negative)
    hit_rate_min: float = 0.30
    avg_payoff_min: float = 1.5
    captured_events_min: int = 5
    shuffle_pvalue_max: float = 0.05
    deflated_sharpe_min: float = 0.0


@dataclass
class Gate1Verdict:
    passed: bool
    failures: list[str] = field(default_factory=list)
    deflated_sharpe_value: float = 0.0


def evaluate_gate1(
    holdout: dict[str, Any],
    n_trials: int,
    n_obs: int,
    thresholds: Gate1Thresholds | None = None,
) -> Gate1Verdict:
    if thresholds is None:
        thresholds = Gate1Thresholds()
    failures: list[str] = []
    if holdout["sharpe"] < thresholds.sharpe_min:
        failures.append(f"sharpe {holdout['sharpe']:.2f} < {thresholds.sharpe_min}")
    if holdout["sortino"] < thresholds.sortino_min:
        failures.append(f"sortino {holdout['sortino']:.2f} < {thresholds.sortino_min}")
    if holdout["max_drawdown"] < thresholds.max_drawdown_max:
        failures.append(
            f"max_drawdown {holdout['max_drawdown']:.2%} < {thresholds.max_drawdown_max:.0%}"
        )
    if holdout["hit_rate"] < thresholds.hit_rate_min:
        failures.append(f"hit_rate {holdout['hit_rate']:.2f} < {thresholds.hit_rate_min}")
    if holdout["avg_payoff"] < thresholds.avg_payoff_min:
        failures.append(f"avg_payoff {holdout['avg_payoff']:.2f} < {thresholds.avg_payoff_min}")
    if (
        holdout["captured_events"] is not None
        and holdout["captured_events"] < thresholds.captured_events_min
    ):
        failures.append(
            f"captured_events {holdout['captured_events']} < {thresholds.captured_events_min}"
        )
    if holdout["shuffle_pvalue"] > thresholds.shuffle_pvalue_max:
        failures.append(
            f"shuffle_pvalue {holdout['shuffle_pvalue']:.3f} > {thresholds.shuffle_pvalue_max}"
        )
    ds = deflated_sharpe(holdout["sharpe"], n_trials=n_trials, n_obs=n_obs)
    if ds < thresholds.deflated_sharpe_min:
        failures.append(f"deflated_sharpe {ds:.2f} < {thresholds.deflated_sharpe_min}")
    return Gate1Verdict(passed=not failures, failures=failures, deflated_sharpe_value=ds)
