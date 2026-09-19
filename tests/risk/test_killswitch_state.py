"""P1/P4 — the sticky-cooldown state machine is one pure function shared by
the backtest runner and the runtime."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from squeeze_hunter.risk.killswitch import KillswitchState, KillSwitchVerdict, advance_killswitch

_T0 = datetime(2026, 5, 14, 14, 0, tzinfo=UTC)


def test_trip_opens_a_sticky_window_and_reports_the_transition() -> None:
    s0 = KillswitchState()
    s1, transition = advance_killswitch(s0, KillSwitchVerdict(True, "monthly_drawdown"), _T0, 7)
    assert s1.active
    assert s1.reason == "monthly_drawdown"
    assert s1.first_tripped_at == _T0
    assert transition == "tripped"
    # Inside the window a clear verdict does not clear the switch.
    s2, t2 = advance_killswitch(s1, KillSwitchVerdict(False), _T0 + timedelta(days=3), 7)
    assert s2.active
    assert t2 is None


def test_window_expiry_follows_the_live_verdict_without_rearming() -> None:
    s0 = KillswitchState(active=True, reason="three_day_loss", first_tripped_at=_T0)
    later = _T0 + timedelta(days=8)
    still_bad, t = advance_killswitch(s0, KillSwitchVerdict(True, "three_day_loss"), later, 7)
    assert still_bad.active
    assert still_bad.first_tripped_at == _T0  # no new window on persistent badness
    assert t is None
    cleared, t2 = advance_killswitch(s0, KillSwitchVerdict(False), later, 7)
    assert not cleared.active
    assert cleared.first_tripped_at is None
    assert t2 == "cleared"
