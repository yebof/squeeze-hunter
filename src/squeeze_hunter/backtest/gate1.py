"""Gate 1 verdict — does the strategy clear all conditions to start paper trading?"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from squeeze_hunter.backtest.deflated_sharpe import deflated_sharpe

if TYPE_CHECKING:
    import pandas as pd

# R9.2: setup-type buckets we expect to see in a fully-validated backtest. If
# any are missing entirely from the trade log, the Gate 1 verdict only
# validates the subset that DID trade, and we surface that as a warning so
# operators don't mistake "PASSED" for "all setups validated."
_EXPECTED_SETUPS = ("CAR", "GME", "Mixed")


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


def resolve_thresholds_for_n_trials(
    n_trials: int, base: Gate1Thresholds | None = None
) -> Gate1Thresholds:
    """Auto-pick deflated_sharpe_min based on n_trials.

    With <=30 trials the multiple-testing penalty is mild, so we want a
    strict threshold (0.3). With more trials the deflated-Sharpe formula
    suppresses the value heavily by design, and a 0.0 threshold makes
    sense (the shuffle test and n_obs penalty already protect us).
    """
    base = base or Gate1Thresholds()
    ds_min = 0.3 if n_trials <= 30 else 0.0
    return Gate1Thresholds(
        sharpe_min=base.sharpe_min,
        sortino_min=base.sortino_min,
        max_drawdown_max=base.max_drawdown_max,
        hit_rate_min=base.hit_rate_min,
        avg_payoff_min=base.avg_payoff_min,
        captured_events_min=base.captured_events_min,
        shuffle_pvalue_max=base.shuffle_pvalue_max,
        deflated_sharpe_min=ds_min,
    )


def evaluate_gate1(
    holdout: dict[str, Any],
    n_trials: int,
    n_obs: int,
    thresholds: Gate1Thresholds | None = None,
) -> Gate1Verdict:
    if thresholds is None:
        thresholds = resolve_thresholds_for_n_trials(n_trials)
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


def setup_type_coverage_warning(trades: pd.DataFrame) -> str | None:
    """R9.2: surface setup-type concentration in the holdout trade log.

    When f4 (Reddit baseline) and f5 (options OI velocity) are not yet wired,
    z[f4] = z[f5] = 0 universally → B = 0 → classifier never produces GME or
    Mixed labels. The "PASSED" verdict in that case validates the CAR setup
    only; operators must be told explicitly so they don't extrapolate to
    untested setups.

    Returns a human-readable warning string (CLI prints it next to the verdict)
    or None when all expected setups appear in the log.
    """
    if trades is None or trades.empty or "setup_type" not in trades.columns:
        return None
    seen = {str(s) for s in trades["setup_type"].dropna().unique()}
    seen &= set(_EXPECTED_SETUPS)
    missing = [s for s in _EXPECTED_SETUPS if s not in seen]
    if not missing:
        return None
    return (
        "Gate 1 trade log only covers setup_type ∈ {"
        + ", ".join(sorted(seen))
        + "}; missing: "
        + ", ".join(missing)
        + ". Verdict validates the covered setups only — see CLAUDE.md notes on "
        "f4 (Reddit baseline) and f5 (options OI velocity) before generalizing."
    )
