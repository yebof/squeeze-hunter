"""PortfolioTelemetry: tracks state for the killswitch.

Tests cover the 5 metrics that feed evaluate_killswitch:
- rolling_30d_max_drawdown
- last_3_days_cumulative_pnl_pct
- worst_position_gap_pct
- broker_disconnected_for_seconds
- critical_data_stale_for_seconds
"""

from datetime import UTC, datetime, timedelta

import pytest

from squeeze_hunter.runtime import PortfolioTelemetry


def _t(d: int, h: int = 12) -> datetime:
    return datetime(2026, 5, d, h, tzinfo=UTC)


def test_drawdown_zero_at_start() -> None:
    tel = PortfolioTelemetry()
    tel.record_equity(_t(1), 100_000.0)
    assert tel.rolling_30d_max_drawdown(_t(1)) == 0.0


def test_drawdown_negative_after_loss() -> None:
    tel = PortfolioTelemetry()
    tel.record_equity(_t(1), 100_000.0)
    tel.record_equity(_t(2), 95_000.0)
    dd = tel.rolling_30d_max_drawdown(_t(2))
    assert dd == pytest.approx(-0.05)


def test_drawdown_uses_running_max() -> None:
    """Peak is 110_000, current is 88_000 → -20% drawdown."""
    tel = PortfolioTelemetry()
    tel.record_equity(_t(1), 100_000.0)
    tel.record_equity(_t(2), 110_000.0)
    tel.record_equity(_t(3), 88_000.0)
    assert tel.rolling_30d_max_drawdown(_t(3)) == pytest.approx(-0.20)


def test_drawdown_window_drops_old_peaks() -> None:
    """A peak 31 days ago should not count toward the rolling window."""
    tel = PortfolioTelemetry()
    tel.record_equity(_t(1), 200_000.0)  # old peak
    # Record many days at 100k inside the 30d window
    for d in range(2, 28):
        tel.record_equity(datetime(2026, 6, d, tzinfo=UTC), 100_000.0)
    tel.record_equity(datetime(2026, 6, 28, tzinfo=UTC), 95_000.0)
    # As of June 28, the May 1 peak (200k) is ~58 days old → out of 30d window.
    # Rolling peak within 30d is 100k. Drawdown = (95k - 100k) / 100k = -0.05.
    dd = tel.rolling_30d_max_drawdown(datetime(2026, 6, 28, tzinfo=UTC))
    assert dd == pytest.approx(-0.05)


def test_three_day_cumulative_pnl() -> None:
    """Sum of returns over the trailing 3 calendar days."""
    tel = PortfolioTelemetry()
    tel.record_equity(_t(1), 100_000.0)
    tel.record_equity(_t(2), 99_000.0)  # -1%
    tel.record_equity(_t(3), 98_000.0)  # -1%
    tel.record_equity(_t(4), 97_000.0)  # -1%
    pnl = tel.last_3_days_cumulative_pnl_pct(_t(4))
    # Cumulative from day 1's start (100k) to day 4's close (97k) = -3%
    # OR sum of daily returns ≈ -3%. Either works as long as test reflects
    # the implementation's semantics.
    assert pnl < 0.0
    assert pnl == pytest.approx(-0.03, rel=0.05)


def test_worst_position_gap_zero_with_no_positions() -> None:
    tel = PortfolioTelemetry()
    assert tel.worst_position_gap_pct() == 0.0


def test_worst_position_gap_finds_largest_drop() -> None:
    """The position with the deepest mark-to-market loss vs entry sets the gap.

    -30% on GME, -10% on AMC → worst is -30%.
    """
    tel = PortfolioTelemetry()
    tel.record_position("GME", entry_price=100.0, mark_price=70.0)
    tel.record_position("AMC", entry_price=10.0, mark_price=9.0)
    assert tel.worst_position_gap_pct() == pytest.approx(-0.30)


def test_worst_position_gap_ignores_winning_positions() -> None:
    tel = PortfolioTelemetry()
    tel.record_position("GME", entry_price=100.0, mark_price=120.0)  # +20%
    assert tel.worst_position_gap_pct() == 0.0


def test_broker_disconnected_zero_when_recently_connected() -> None:
    tel = PortfolioTelemetry()
    tel.record_broker_heartbeat(_t(1, 12))
    assert tel.broker_disconnected_for_seconds(_t(1, 12)) == 0
    assert tel.broker_disconnected_for_seconds(_t(1, 12)) == 0


def test_broker_disconnected_counts_seconds_since_last_heartbeat() -> None:
    tel = PortfolioTelemetry()
    tel.record_broker_heartbeat(datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC))
    later = datetime(2026, 5, 1, 12, 5, 30, tzinfo=UTC)  # 5min30s later
    assert tel.broker_disconnected_for_seconds(later) == 330


def test_data_stale_zero_when_recent() -> None:
    tel = PortfolioTelemetry()
    tel.record_data_freshness("ibkr_quotes", _t(1, 12))
    assert tel.critical_data_stale_for_seconds(_t(1, 12)) == 0


def test_data_stale_uses_oldest_critical_source() -> None:
    """If 'ibkr_quotes' was 1h ago and 'finra' was 25h ago, the staleness is 25h.

    But finra is a slow-changing source — the test should distinguish 'critical'
    sources (quotes, options) from 'slow' sources (FINRA, earnings calendar).
    Implementation choice: a `critical_sources` set on telemetry.
    """
    tel = PortfolioTelemetry(critical_sources={"ibkr_quotes"})
    tel.record_data_freshness("ibkr_quotes", datetime(2026, 5, 1, 11, tzinfo=UTC))
    tel.record_data_freshness("finra", datetime(2026, 4, 30, tzinfo=UTC))
    later = datetime(2026, 5, 1, 13, tzinfo=UTC)  # 2h after ibkr_quotes
    assert tel.critical_data_stale_for_seconds(later) == 2 * 60 * 60  # 7200s


def test_data_freshness_unconditional_recording() -> None:
    """R5.C3 regression: data_freshness should be updatable independent of
    whether positions exist. Previously runtime.tick() only called
    record_data_freshness inside the position loop, so an empty portfolio
    meant the timestamp never updated → after 2 hours of flat portfolio the
    data_stale killswitch arm tripped on a healthy broker. The runtime fix
    moves the record outside the position loop; this unit test confirms the
    telemetry API supports that usage.
    """
    tel = PortfolioTelemetry()
    # Fresh recording with no positions
    tel.record_data_freshness("ibkr_quotes", _t(1))
    assert tel.critical_data_stale_for_seconds(_t(1)) == 0
    # 2 hours later without re-recording, staleness grows correctly
    later = _t(1) + timedelta(hours=2)
    assert tel.critical_data_stale_for_seconds(later) == 7200


def test_drawdown_preserves_intraday_peak_after_dedupe_by_day() -> None:
    """R7.I1: R6.C1's dedupe-by-day (last write wins) would otherwise erase an
    intraday equity peak. Drawdown math should still see the peak via the
    parallel equity_peak_per_day tracker.
    """
    tel = PortfolioTelemetry()
    # Day 1: ramp from 100k up to 110k, then close back at 100k. The latest
    # entry by-day will overwrite to 100k, but the peak should still be 110k.
    tel.record_equity(datetime(2026, 5, 11, 10, 0, tzinfo=UTC), 100_000.0)
    tel.record_equity(datetime(2026, 5, 11, 12, 0, tzinfo=UTC), 110_000.0)
    tel.record_equity(datetime(2026, 5, 11, 16, 0, tzinfo=UTC), 100_000.0)
    # Day 2: equity drops to 99k. Drawdown vs the 110k peak should be -10%.
    tel.record_equity(datetime(2026, 5, 12, 16, 0, tzinfo=UTC), 99_000.0)
    dd = tel.rolling_30d_max_drawdown(datetime(2026, 5, 12, 17, tzinfo=UTC))
    assert dd == pytest.approx(-0.10, abs=0.001)


def test_three_day_pnl_uses_trading_days_not_calendar() -> None:
    """R7.I2: Monday-morning call should look back 3 trading days (Wed/Thu/Fri
    of previous week), not 3 calendar days (Sat/Sun/Mon) which would yield
    1 trading day of history at best.
    """
    tel = PortfolioTelemetry()
    # The trailing 3 trading days before Monday 2026-05-11 are Wed 5/6, Thu 5/7,
    # Fri 5/8. Seed equity values across those days plus the current Monday.
    tel.record_equity(datetime(2026, 5, 6, 16, tzinfo=UTC), 100_000.0)  # Wed
    tel.record_equity(datetime(2026, 5, 7, 16, tzinfo=UTC), 99_000.0)  # Thu
    tel.record_equity(datetime(2026, 5, 8, 16, tzinfo=UTC), 98_000.0)  # Fri
    tel.record_equity(datetime(2026, 5, 11, 16, tzinfo=UTC), 95_000.0)  # Mon
    pnl = tel.last_3_days_cumulative_pnl_pct(
        datetime(2026, 5, 11, 16, 30, tzinfo=UTC),
    )
    # 95k vs 100k → -5%, far below the -5% killswitch threshold. With the old
    # calendar-day cutoff (Sat-Sun-Mon), only one entry would be in the
    # window and the metric would return 0.0 → killswitch dead on Mondays.
    assert pnl == pytest.approx(-0.05, abs=0.005)


def test_telemetry_to_killswitch_inputs() -> None:
    """Convenience: telemetry can produce KillSwitchInputs in one call."""
    from squeeze_hunter.risk.killswitch import KillSwitchInputs

    tel = PortfolioTelemetry()
    tel.record_equity(_t(1), 100_000.0)
    tel.record_broker_heartbeat(_t(1))
    tel.record_data_freshness("ibkr_quotes", _t(1))
    inputs = tel.to_killswitch_inputs(as_of=_t(1))
    assert isinstance(inputs, KillSwitchInputs)
    assert inputs.rolling_30d_max_drawdown == 0.0
    assert inputs.broker_disconnected_for_seconds == 0
