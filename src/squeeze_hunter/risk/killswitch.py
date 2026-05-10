"""Killswitch — pure function over telemetry inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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
    if inp.broker_disconnected_for_seconds > broker_outage_max_seconds:
        return KillSwitchVerdict(True, "broker_outage")
    if inp.critical_data_stale_for_seconds > data_stale_max_seconds:
        return KillSwitchVerdict(True, "data_stale")
    return KillSwitchVerdict(False)
