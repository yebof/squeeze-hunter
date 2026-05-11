from datetime import UTC, datetime

from squeeze_hunter.risk.killswitch import KillSwitchInputs, evaluate_killswitch


def _base() -> KillSwitchInputs:
    return KillSwitchInputs(
        as_of=datetime(2026, 5, 14, 14, 0, tzinfo=UTC),
        rolling_30d_max_drawdown=-0.05,
        last_3_days_cumulative_pnl_pct=-0.02,
        worst_position_gap_pct=-0.10,
        broker_disconnected_for_seconds=0,
        critical_data_stale_for_seconds=0,
    )


def test_no_trigger_baseline() -> None:
    v = evaluate_killswitch(_base())
    assert not v.tripped


def test_monthly_drawdown_trips() -> None:
    inp = _base()
    inp.rolling_30d_max_drawdown = -0.11
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "monthly_drawdown"


def test_three_day_loss_trips() -> None:
    inp = _base()
    inp.last_3_days_cumulative_pnl_pct = -0.06
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "three_day_loss"


def test_gap_through_stop_trips() -> None:
    inp = _base()
    inp.worst_position_gap_pct = -0.30
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "gap_through_stop"


def test_broker_outage_trips() -> None:
    inp = _base()
    inp.broker_disconnected_for_seconds = 400
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "broker_outage"


def test_data_stale_trips() -> None:
    inp = _base()
    inp.critical_data_stale_for_seconds = 60 * 60 * 3
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "data_stale"


def test_broker_outage_at_exact_threshold_trips() -> None:
    """R7.M3: spec reads 'at or above 5 minutes' — comparison must be >=, not >.

    Before R7.M3: 300s exactly returned tripped=False, leaving a 1-second
    window where the broker outage trigger silently disarmed.
    """
    inp = _base()
    inp.broker_disconnected_for_seconds = 300  # exactly at threshold
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "broker_outage"


def test_data_stale_at_exact_threshold_trips() -> None:
    """R7.M3: same logic for data_stale threshold."""
    inp = _base()
    inp.critical_data_stale_for_seconds = 60 * 60 * 2  # exactly 2h
    v = evaluate_killswitch(inp)
    assert v.tripped
    assert v.reason == "data_stale"
