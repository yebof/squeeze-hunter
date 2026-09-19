"""Killswitch controller (P4): the sticky-cooldown state plus its side effects.

`evaluate_killswitch` (pure verdict) and `advance_killswitch` (pure state
step) stay in `risk/killswitch.py`; this object owns the mutable state that
used to be five loose fields on RuntimeContext, and the gauge / reason
bookkeeping around it. Alerts and logging are the caller's business — the
controller only reports the transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from squeeze_hunter.config import Settings
from squeeze_hunter.monitor.metrics import MetricsRegistry
from squeeze_hunter.risk.killswitch import (
    KillSwitchInputs,
    KillswitchState,
    advance_killswitch,
    evaluate_killswitch,
)


@dataclass
class KillswitchController:
    cooldown_days: int = 7
    active: bool = False
    reason: str | None = None
    first_tripped_at: datetime | None = None
    # R10.2: every reason label set during a trip cycle, so the eventual
    # clear resets ALL of them (the trip reason can transition mid-cycle and
    # Prometheus tracks each label separately).
    active_reasons: set[str] = field(default_factory=set)

    def evaluate(
        self,
        inputs: KillSwitchInputs,
        settings: Settings,
        now: datetime,
        *,
        metrics: MetricsRegistry | None,
    ) -> Literal["tripped", "cleared"] | None:
        """Run the verdict and the sticky-cooldown step; update gauges.
        Returns the transition ("tripped" / "cleared") or None."""
        ks_cfg = settings.risk.killswitch
        verdict = evaluate_killswitch(
            inputs,
            # R8.S-I2: YAML stores the drawdown magnitude positive.
            monthly_drawdown_max=-abs(settings.risk.monthly_drawdown_kill),
            three_day_loss_max=ks_cfg.three_day_loss_max,
            gap_through_stop_max=ks_cfg.gap_through_stop_max,
            broker_outage_max_seconds=ks_cfg.broker_outage_max_seconds,
            data_stale_max_seconds=ks_cfg.data_stale_max_seconds,
        )
        prior = KillswitchState(self.active, self.reason, self.first_tripped_at)
        nxt, transition = advance_killswitch(prior, verdict, now, self.cooldown_days)
        self.active, self.reason, self.first_tripped_at = (
            nxt.active,
            nxt.reason,
            nxt.first_tripped_at,
        )
        if nxt.active:
            if metrics is not None:
                label = nxt.reason or "unknown"
                metrics.set_kill_switch_active(label)
                self.active_reasons.add(label)
        elif transition == "cleared":
            self._clear_gauges(metrics)
        return transition

    def reset(self, *, metrics: MetricsRegistry | None) -> None:
        """R7.C1: explicit manual reset — clears the sticky window even if the
        cooldown has not elapsed. The operator is responsible for verifying
        that the triggering condition has actually resolved."""
        self._clear_gauges(metrics)
        self.first_tripped_at = None
        self.active = False
        self.reason = None

    def _clear_gauges(self, metrics: MetricsRegistry | None) -> None:
        # R9.4 + R10.2: reset every label set during the cycle so Grafana
        # shows all gauges back at 0.0.
        if metrics is not None:
            for label in self.active_reasons:
                metrics.set_kill_switch_inactive(label)
        self.active_reasons.clear()

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "reason": self.reason,
            "first_tripped_at": self.first_tripped_at.isoformat()
            if self.first_tripped_at
            else None,
            "active_reasons": sorted(self.active_reasons),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.active = bool(snap.get("active", False))
        self.reason = snap.get("reason")
        raw = snap.get("first_tripped_at")
        self.first_tripped_at = datetime.fromisoformat(raw) if raw else None
        self.active_reasons = set(snap.get("active_reasons") or [])
