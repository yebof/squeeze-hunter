"""Killswitch — pure function over telemetry inputs, plus the sticky-cooldown
state machine shared by the backtest runner and the runtime (P1/P4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal


@dataclass
class KillSwitchInputs:
    as_of: datetime
    rolling_30d_max_drawdown: float  # negative number, e.g. -0.10 = -10%
    last_3_days_cumulative_pnl_pct: float
    worst_position_gap_pct: float  # most negative single-position gap
    broker_disconnected_for_seconds: int
    critical_data_stale_for_seconds: int


@dataclass(slots=True, frozen=True)
class KillSwitchVerdict:
    tripped: bool
    reason: str | None = None


def evaluate_killswitch(
    inp: KillSwitchInputs,
    *,
    monthly_drawdown_max: float = -0.10,
    three_day_loss_max: float = -0.05,
    gap_through_stop_max: float = -0.25,
    broker_outage_max_seconds: int = 300,
    data_stale_max_seconds: int = 60 * 60 * 2,
) -> KillSwitchVerdict:
    if inp.rolling_30d_max_drawdown <= monthly_drawdown_max:
        return KillSwitchVerdict(True, "monthly_drawdown")
    if inp.last_3_days_cumulative_pnl_pct <= three_day_loss_max:
        return KillSwitchVerdict(True, "three_day_loss")
    if inp.worst_position_gap_pct <= gap_through_stop_max:
        return KillSwitchVerdict(True, "gap_through_stop")
    # R7.M3: >= so the spec's "at or above 5 minutes" / "at or above 2 hours"
    # readings actually trip at the stated threshold, not one second past it.
    if inp.broker_disconnected_for_seconds >= broker_outage_max_seconds:
        return KillSwitchVerdict(True, "broker_outage")
    if inp.critical_data_stale_for_seconds >= data_stale_max_seconds:
        return KillSwitchVerdict(True, "data_stale")
    return KillSwitchVerdict(False)


@dataclass(frozen=True, slots=True)
class KillswitchState:
    active: bool = False
    reason: str | None = None
    first_tripped_at: datetime | None = None


def advance_killswitch(
    state: KillswitchState,
    verdict: KillSwitchVerdict,
    now: datetime,
    cooldown_days: int,
) -> tuple[KillswitchState, Literal["tripped", "cleared"] | None]:
    """R7.C1 + R8.C1 sticky cooldown, as one pure step.

    From the first trip the switch stays active for `cooldown_days` calendar
    days regardless of the live verdict. After the window it follows the
    verdict but does NOT open a new window on persistent badness — only a
    fresh clear → tripped transition does. Returns the new state and the
    transition that happened this step (for logging / alerting / gauges).
    """
    if state.first_tripped_at is not None and now < state.first_tripped_at + timedelta(
        days=cooldown_days
    ):
        return KillswitchState(True, state.reason or "cooldown", state.first_tripped_at), None
    if verdict.tripped:
        if not state.active:
            return KillswitchState(True, verdict.reason, now), "tripped"
        return KillswitchState(True, verdict.reason, state.first_tripped_at), None
    if state.first_tripped_at is not None or state.active:
        return KillswitchState(False, None, None), "cleared"
    return KillswitchState(False, None, None), None
