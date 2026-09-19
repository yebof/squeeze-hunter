"""Portfolio telemetry: the five killswitch inputs, derived from recorded
history. P4 slice: moved out of runtime.py so the backtest runner can feed
the SAME class (one killswitch-input derivation for backtest and live)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from squeeze_hunter.risk.killswitch import KillSwitchInputs


@dataclass
class PortfolioTelemetry:
    """Tracks the inputs that feed evaluate_killswitch.

    All metrics are computed lazily on demand from recorded history.
    """

    equity_history: list[tuple[datetime, float]] = field(default_factory=list)
    # R7.I1: parallel tracker for the per-day MAX equity. equity_history
    # collapses to last-of-day for the 3-day PnL metric (which wants
    # close-vs-close), but the drawdown calc needs intraday peaks. Two
    # accumulators keep both metrics correct without conflating them.
    equity_peak_per_day: dict[date, float] = field(default_factory=dict)
    position_marks: dict[str, tuple[float, float]] = field(default_factory=dict)
    # ticker -> (entry_price, mark_price); negative gap = adverse move
    last_broker_heartbeat: datetime | None = None
    data_freshness: dict[str, datetime] = field(default_factory=dict)
    critical_sources: set[str] = field(default_factory=lambda: {"ibkr_quotes"})

    def record_equity(self: PortfolioTelemetry, ts: datetime, equity_usd: float) -> None:
        # R6.C1: dedupe by date — keep only the LATEST equity reading per
        # calendar day (in UTC). At 60s intraday ticks this collapses ~390
        # entries/day into 1, so over 30 days we hold ≤31 entries — the trim
        # is then trivial. The previous R4.9 implementation tried to trim by
        # 31-day cutoff after len>100, but in a single trading session ALL
        # entries were within the cutoff → no actual trim → unbounded growth.
        ts_date = ts.date()
        if self.equity_history and self.equity_history[-1][0].date() == ts_date:
            # Same day as the last entry — replace it with the newer mark
            self.equity_history[-1] = (ts, equity_usd)
        else:
            self.equity_history.append((ts, equity_usd))
        # R7.I1: separately track the per-day max so the drawdown metric does
        # not lose intraday peaks when a later same-day overwrite is lower.
        prior_peak = self.equity_peak_per_day.get(ts_date, equity_usd)
        self.equity_peak_per_day[ts_date] = max(prior_peak, equity_usd)
        # Cap to last 60 entries (~60 trading days). All metrics use at most
        # a 30-day window, so 60 entries is a generous ceiling.
        if len(self.equity_history) > 60:
            self.equity_history = self.equity_history[-60:]
        # R8.Q-I9: evict equity_peak_per_day entries strictly older than 90
        # CALENDAR DAYS from this `ts`, not just by count. After a long
        # outage the count-only cap would retain entries from months ago
        # because the dict was below the threshold.
        stale_cutoff = ts_date - timedelta(days=90)
        self.equity_peak_per_day = {
            d: p for d, p in self.equity_peak_per_day.items() if d >= stale_cutoff
        }

    def record_position(
        self: PortfolioTelemetry, ticker: str, entry_price: float, mark_price: float
    ) -> None:
        self.position_marks[ticker] = (entry_price, mark_price)

    def clear_position(self: PortfolioTelemetry, ticker: str) -> None:
        self.position_marks.pop(ticker, None)

    def record_broker_heartbeat(self: PortfolioTelemetry, ts: datetime) -> None:
        self.last_broker_heartbeat = ts

    def record_data_freshness(self: PortfolioTelemetry, source: str, ts: datetime) -> None:
        self.data_freshness[source] = ts

    def rolling_30d_max_drawdown(self: PortfolioTelemetry, as_of: datetime) -> float:
        cutoff = as_of - timedelta(days=30)
        cutoff_date = cutoff.date()
        # R8.S-I5: read the per-day peak tracker FIRST so a same-day drawdown
        # (Day 1 of operation: peak intraday + adverse close) is detected.
        # Prior order required len(equity_history) >= 2 first, returning 0
        # for single-day drawdowns even though equity_peak_per_day had today's
        # high recorded.
        peaks_in_window = [e for d, e in self.equity_peak_per_day.items() if d >= cutoff_date]
        if self.equity_history:
            current = self.equity_history[-1][1]
        else:
            return 0.0
        if peaks_in_window:
            peak = max(peaks_in_window)
        else:
            # Parallel tracker empty (legacy state): fall back to equity_history.
            recent = [(t, e) for t, e in self.equity_history if t >= cutoff]
            if len(recent) < 2:
                return 0.0
            peak = max(e for _, e in recent)
        if peak <= 0:
            return 0.0
        return (current - peak) / peak

    def last_3_days_cumulative_pnl_pct(self: PortfolioTelemetry, as_of: datetime) -> float:
        # R7.I2: 3 *trading* days, not calendar days. On a Monday the calendar
        # cutoff would only include Sat+Sun+today → at most 1 equity entry,
        # and the metric silently returns 0 (killswitch dead on Mondays).
        # Use a 4-business-day window ending at as_of so we capture the prior
        # 3 trading days plus today.
        bdays = pd.bdate_range(end=as_of, periods=4)
        cutoff = bdays[0].to_pydatetime().replace(tzinfo=UTC)
        recent = [(t, e) for t, e in self.equity_history if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        start_equity = recent[0][1]
        end_equity = recent[-1][1]
        if start_equity <= 0:
            return 0.0
        return (end_equity - start_equity) / start_equity

    def worst_position_gap_pct(self: PortfolioTelemetry) -> float:
        if not self.position_marks:
            return 0.0
        worst = 0.0
        for entry, mark in self.position_marks.values():
            if entry <= 0:
                continue
            gap = (mark - entry) / entry
            if gap < worst:
                worst = gap
        return worst

    def clamp_freshness_to(self: PortfolioTelemetry, floor: datetime) -> None:
        """Raise heartbeat / data-freshness stamps older than `floor` up to it.

        Round-12: the outage arms must measure IN-SESSION time. Ticks outside
        09:30-16:00 ET return before touching telemetry, so both stamps froze
        at ~15:59 ET; a single transient health() failure at the next open
        then read as a 17 h (65 h over a weekend) outage → 7-day lockout.
        """
        if self.last_broker_heartbeat is not None and self.last_broker_heartbeat < floor:
            self.last_broker_heartbeat = floor
        for src, ts in list(self.data_freshness.items()):
            if ts < floor:
                self.data_freshness[src] = floor

    def broker_disconnected_for_seconds(self: PortfolioTelemetry, as_of: datetime) -> int:
        if self.last_broker_heartbeat is None:
            return 0
        delta = as_of - self.last_broker_heartbeat
        return max(0, int(delta.total_seconds()))

    def critical_data_stale_for_seconds(self: PortfolioTelemetry, as_of: datetime) -> int:
        relevant = [ts for src, ts in self.data_freshness.items() if src in self.critical_sources]
        if not relevant:
            return 0
        oldest = min(relevant)
        delta = as_of - oldest
        return max(0, int(delta.total_seconds()))

    def to_killswitch_inputs(self: PortfolioTelemetry, as_of: datetime) -> KillSwitchInputs:
        from squeeze_hunter.risk.killswitch import KillSwitchInputs

        return KillSwitchInputs(
            as_of=as_of,
            rolling_30d_max_drawdown=self.rolling_30d_max_drawdown(as_of),
            last_3_days_cumulative_pnl_pct=self.last_3_days_cumulative_pnl_pct(as_of),
            worst_position_gap_pct=self.worst_position_gap_pct(),
            broker_disconnected_for_seconds=self.broker_disconnected_for_seconds(as_of),
            critical_data_stale_for_seconds=self.critical_data_stale_for_seconds(as_of),
        )


# R7.I3: transient broker/data errors caught around per-tick I/O. Programming
# errors (AttributeError, NotImplementedError, TypeError, ValueError) must
# propagate up to tick_safe so they surface in structured logs.
