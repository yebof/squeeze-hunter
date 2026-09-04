"""Top-level runtime — wires settings, broker, scheduler, monitor into one process."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.base import IBroker
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.execution.lifecycle import LifecycleState, manage_positions
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.monitor.alerts import AlertSender, Severity
from squeeze_hunter.monitor.http import MonitorServer, start_monitor_server
from squeeze_hunter.monitor.metrics import MetricsRegistry
from squeeze_hunter.risk.killswitch import evaluate_killswitch

if TYPE_CHECKING:
    from squeeze_hunter.risk.killswitch import KillSwitchInputs

log = get_logger("runtime")

# Alert delivery is best-effort: transport errors are logged, never raised.
_ALERT_ERRORS = (httpx.HTTPError, OSError, TimeoutError)


def _alert_sender_from_env() -> AlertSender | None:
    """Build the alert channel from TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID and
    SLACK_WEBHOOK_URL; None when nothing is configured (logged on first use)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or None
    chat = os.environ.get("TELEGRAM_CHAT_ID") or None
    slack = os.environ.get("SLACK_WEBHOOK_URL") or None
    if not ((token and chat) or slack):
        return None
    return AlertSender(telegram_bot_token=token, telegram_chat_id=chat, slack_webhook_url=slack)


# R3.2: US regular session for the intraday loop. 09:30-16:00 ET, Mon-Fri.
# We do NOT filter US federal holidays here — the cost of trading on a
# holiday (rare false positive) is a logged debug skip, and we'd rather
# err on the side of being available than silently skip a half-day session
# (which is unusual but legal). Holiday-exact filtering can be added if needed.
_NY = ZoneInfo("America/New_York")
_SESSION_OPEN = time(9, 30)
_SESSION_CLOSE = time(16, 0)


def _is_us_regular_session(now: datetime) -> bool:
    """True if `now` falls within Mon-Fri 09:30-16:00 ET (regular trading)."""
    et = now.astimezone(_NY)
    if et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return _SESSION_OPEN <= et.time() < _SESSION_CLOSE


def _session_open_utc(now: datetime) -> datetime:
    """Today's 09:30 ET (for `now`'s ET calendar day) expressed in UTC."""
    et = now.astimezone(_NY)
    return datetime.combine(et.date(), _SESSION_OPEN, tzinfo=_NY).astimezone(UTC)


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
_TRANSIENT_IO_ERRORS = (ConnectionError, TimeoutError, OSError)


@dataclass
class RuntimeContext:
    cache: ParquetCache
    settings: Settings
    tickers: list[str]
    mode: str = "paper"  # "paper" | "live" | "sim"
    broker: IBroker | None = None
    metrics_registry: MetricsRegistry | None = None
    lifecycle_state: LifecycleState = field(default_factory=LifecycleState)
    kill_switch_active: bool = False
    _kill_reason: str | None = None
    # R7.C1: sticky-cooldown bookkeeping. Once tripped, the killswitch stays
    # tripped for `_kill_cooldown_days` calendar days regardless of fresh
    # telemetry. Spec: "Auto-resume after 7 calendar days OR explicit manual
    # reset." Prior behavior recomputed every tick → would untrip the same
    # tick the triggering input recovered.
    _kill_first_tripped_at: datetime | None = None
    _kill_cooldown_days: int = 7
    # R10.2: track every distinct reason label that has been set on the
    # killswitch Prometheus gauge during the current trip cycle. Prometheus
    # gauges with labels are SEPARATE time series per label value, so when the
    # trip reason transitions (e.g., monthly_drawdown → data_stale) we leave
    # the prior label's gauge stuck at 1.0 unless we explicitly reset all of
    # them on clear. Repopulated each cycle; cleared on auto-clear / manual
    # reset together with the gauges themselves.
    _active_kill_reasons: set[str] = field(default_factory=set)
    telemetry: PortfolioTelemetry = field(default_factory=PortfolioTelemetry)
    # Populated by nightly_scan; read by premarket_verify the next morning.
    last_candidates: pd.DataFrame | None = None
    # R9.7: top-level guard against overlapping ticks. APScheduler's
    # IntervalTrigger fires every 60s; the lambda dispatches via create_task
    # and returns immediately, so a tick that takes >60s would otherwise be
    # joined by a parallel tick. LifecycleState.lock only serializes per-ticker
    # stop processing — broker.health, equity fetch, killswitch evaluation, and
    # telemetry recording would all race. We early-return when a tick is
    # already in flight; the next interval will catch up.
    _tick_in_progress: bool = False
    # Round-12: last known broker health (served by /health), the alert
    # channel (from env) and the /metrics + /health server — wired in setup().
    last_broker_healthy: bool = False
    alerts: AlertSender | None = None
    monitor_server: MonitorServer | None = None

    async def setup(self: RuntimeContext, connect_timeout_s: float = 30.0) -> None:
        # R9.1: propagate YAML stops settings into the lifecycle state so the
        # paper/live stop evaluation uses the same thresholds the operator
        # configures in YAML. Trailing values are stored negative in YAML;
        # evaluate_stops expects positive magnitudes — convert with abs().
        self.lifecycle_state.hard_stop = self.settings.stops.hard_stop
        self.lifecycle_state.time_stop_bars = self.settings.stops.time_stop_days
        self.lifecycle_state.signal_decay_halve = self.settings.stops.signal_decay_halve
        self.lifecycle_state.signal_decay_exit = self.settings.stops.signal_decay_exit
        self.lifecycle_state.trailing_car = abs(self.settings.stops.trailing_car)
        self.lifecycle_state.trailing_gme = abs(self.settings.stops.trailing_gme)
        self.lifecycle_state.trailing_mixed = abs(self.settings.stops.trailing_mixed)
        if self.broker is None:
            if self.mode == "sim":
                self.broker = cast(
                    IBroker,
                    SimulatorBroker(
                        initial_cash=100_000.0,
                        cost_model=StockCostModel(),
                    ),
                )
            elif self.mode == "paper":
                from squeeze_hunter.broker.paper import PaperBroker

                self.broker = PaperBroker(client_id=int(os.environ.get("IBKR_CLIENT_ID", "42")))
                # R3.3: bound the connect attempt so a hung TWS doesn't freeze
                # the process forever with no diagnostic.
                # R4.4: clean up the broker on TimeoutError so we don't leak
                # a partial socket connection. Re-raise so the supervisor
                # learns about the failure.
                try:
                    await asyncio.wait_for(self.broker.connect(), timeout=connect_timeout_s)
                except TimeoutError:
                    log.error("paper_broker_connect_timeout", timeout_s=connect_timeout_s)
                    await self._try_cleanup_partial_broker()
                    raise
            elif self.mode == "live":
                from squeeze_hunter.broker.ibkr import IBKRBroker

                self.broker = IBKRBroker(client_id=int(os.environ.get("IBKR_CLIENT_ID", "42")))
                try:
                    await asyncio.wait_for(self.broker.connect(), timeout=connect_timeout_s)
                except TimeoutError:
                    log.error("live_broker_connect_timeout", timeout_s=connect_timeout_s)
                    await self._try_cleanup_partial_broker()
                    raise
            else:
                raise ValueError(f"unknown mode: {self.mode}")
        self.metrics_registry = MetricsRegistry()
        # R7: seed the broker heartbeat at setup so broker_disconnected_for_seconds
        # measures elapsed time correctly. Without this seed, a startup where the
        # broker is unreachable shows 0 disconnect-seconds forever (the killswitch
        # never trips because last_broker_heartbeat stays None).
        try:
            health = await self.broker.health()
            if health.connected:
                self.telemetry.record_broker_heartbeat(datetime.now(UTC))
        except _TRANSIENT_IO_ERRORS as e:
            # R7.I3: narrow to transient errors at setup. Programming bugs
            # (AttributeError etc.) should propagate so setup() callers see them.
            log.warning(
                "broker_health_unreachable_at_setup",
                err=str(e),
                err_type=type(e).__name__,
            )
            # Seed heartbeat anyway so disconnect timer measures from now.
            # If still down at next tick, broker_disconnected_for_seconds grows.
            self.telemetry.record_broker_heartbeat(datetime.now(UTC))
        # Round-12: alert channel (AlertSender had no caller — killswitch trips
        # only logged) and the /metrics + /health endpoint (Prometheus scraped
        # an empty port). Both were promised by spec §7.
        self.alerts = _alert_sender_from_env()
        if self.settings.monitor.http_port > 0 and self.monitor_server is None:
            self.monitor_server = start_monitor_server(
                self,
                port=self.settings.monitor.http_port,
                host=self.settings.monitor.http_host,
            )

    async def _notify(self: RuntimeContext, text: str) -> None:
        """Push a HIGH-severity alert; delivery failure must never break a tick."""
        if self.alerts is None:
            log.warning("alert_channel_not_configured", text=text)
            return
        try:
            await self.alerts.send(text, severity=Severity.HIGH)
        except _ALERT_ERRORS as e:
            log.warning("alert_send_failed", err=str(e), err_type=type(e).__name__)

    async def tick(self: RuntimeContext, now: datetime) -> None:
        """One intraday tick: manage positions + check killswitch.

        R3.2: skips work outside US regular trading hours (Mon-Fri 09:30-16:00 ET).
        Stops, mark-to-market, and killswitch evaluation only run during the
        regular session — preventing after-hours market orders from auto-exits
        that would otherwise route through AH liquidity (3-8% adverse slippage
        on thin stocks).
        """
        if self.broker is None or self.metrics_registry is None:
            raise RuntimeError("setup() not called")
        # R8.Q-I1: normalize tz-naive `now` to UTC so downstream comparisons
        # against tz-aware `_kill_first_tripped_at` and tz-aware telemetry
        # timestamps don't raise TypeError. Production schedules supply UTC,
        # but tests or future schedulers may pass naive.
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        if not _is_us_regular_session(now):
            log.debug("tick_skipped_outside_session", now=now.isoformat())
            return

        # R9.7: top-level mutex. If the prior tick is still running, skip this
        # one — APScheduler will fire the next interval. We check-and-set
        # synchronously (no await between read and write) so two simultaneous
        # awaits don't both observe False. The flag is reset in `finally` so
        # an exception in the body (re-raised to tick_safe) still releases it.
        if self._tick_in_progress:
            log.warning("tick_overlapped_skipped", now=now.isoformat())
            return
        self._tick_in_progress = True
        try:
            await self._tick_body(now=now)
        finally:
            self._tick_in_progress = False

    async def _tick_body(self: RuntimeContext, now: datetime) -> None:
        """The real work of one tick. Wrapped by `tick()` with the
        `_tick_in_progress` mutex (R9.7). Tests can monkey-patch this to
        observe whether the body actually ran when ticks overlap.
        """
        # broker / metrics_registry already validated by tick()'s entry guard.
        assert self.broker is not None
        assert self.metrics_registry is not None
        # Round-12: outage timers count from today's open, not from the last
        # in-session tick of the previous day (see clamp_freshness_to).
        self.telemetry.clamp_freshness_to(_session_open_utc(now))
        # R3.1: capture position keys BEFORE manage_positions so we can detect
        # which positions were exited this tick and clear their stale telemetry
        # marks. Without this, worst_position_gap_pct keeps reading the last
        # mark of exited positions forever, permanently arming the killswitch
        # gap-through-stop trigger.
        positions_before = set(self.lifecycle_state.positions.keys())
        await manage_positions(self.lifecycle_state, self.broker, now)
        positions_after = set(self.lifecycle_state.positions.keys())
        for exited in positions_before - positions_after:
            self.telemetry.clear_position(exited)

        # Update broker heartbeat if broker is still responsive.
        broker_healthy = False
        try:
            health = await self.broker.health()
            if health.connected:
                self.telemetry.record_broker_heartbeat(now)
                broker_healthy = True
        except _TRANSIENT_IO_ERRORS as e:
            # R7.I3: narrow to transient errors. AttributeError / TypeError
            # etc. are real bugs and must propagate to tick_safe.
            log.warning("broker_health_check_failed", err=str(e), err_type=type(e).__name__)

        # R5.C3: record data freshness UNCONDITIONALLY when the broker
        # responded — not only inside the position loop. Previously, an empty
        # portfolio meant ibkr_quotes never updated, so after 2 hours of flat
        # portfolio the data_stale killswitch arm tripped on a healthy broker.
        # Now: any successful broker.health() round-trip refreshes the source.
        if broker_healthy:
            self.telemetry.record_data_freshness("ibkr_quotes", now)
        self.last_broker_healthy = broker_healthy

        # Mark to market: fetch quotes, record position marks, collect prices.
        # R7.Q-I2: skip the loop entirely when the broker is unhealthy — every
        # fetch_quote will likely raise the same connection issue. Compounds
        # tick latency for no benefit; quotes will refresh next tick anyway.
        marks: dict[str, float] = {}
        if broker_healthy:
            for ticker, meta in self.lifecycle_state.positions.items():
                try:
                    q = await self.broker.fetch_quote(ticker)
                except _TRANSIENT_IO_ERRORS as e:
                    # R7.I3: narrow exception; programming errors propagate.
                    log.debug(
                        "quote_fetch_transient",
                        ticker=ticker,
                        err=str(e),
                        err_type=type(e).__name__,
                    )
                    continue
                price = q.last or q.bid or q.ask
                if price > 0:
                    marks[ticker] = price
                    self.telemetry.record_position(ticker, meta["entry_price"], price)
                    self.telemetry.record_data_freshness("ibkr_quotes", now)

        # Update broker equity with current marks and record for telemetry.
        # Mark-to-market the simulator first so its `equity` field is fresh
        # before we query it through the unified get_equity_usd path.
        if isinstance(self.broker, SimulatorBroker):
            self.broker.mark_to_market(marks, now)

        # R4.1: query equity through IBroker.get_equity_usd in ALL modes.
        # R5.M2: record even when negative — a leveraged account NAV<0 is
        # exactly the catastrophic case the drawdown killswitch should catch.
        # Only skip when the broker returns None (snapshot not yet available).
        try:
            equity_usd = await self.broker.get_equity_usd()
        except _TRANSIENT_IO_ERRORS as e:
            # R7.I3: narrow exception; programming errors propagate.
            log.warning("equity_fetch_failed", err=str(e), err_type=type(e).__name__)
            equity_usd = None
        if equity_usd is not None:
            # R6.C1 made record_equity dedupe-by-day; safe to call every tick.
            self.telemetry.record_equity(now, equity_usd)
            # R6.I3: wire equity and basic state into the Prometheus metrics
            # registry. Most metric families were defined but never updated,
            # so Grafana dashboards would have shown all zeros.
            if self.metrics_registry is not None:
                self.metrics_registry.set_equity(equity_usd)
                self.metrics_registry.position_count.set(len(self.lifecycle_state.positions))
                self.metrics_registry.broker_connected.set(1.0 if broker_healthy else 0.0)

        # Build killswitch inputs from real telemetry and evaluate.
        # R8.S-I2: pass YAML monthly_drawdown_kill so the YAML knob is wired.
        # The YAML value is positive magnitude; killswitch expects negative.
        monthly_dd_kill = -abs(self.settings.risk.monthly_drawdown_kill)
        ks = evaluate_killswitch(
            self.telemetry.to_killswitch_inputs(as_of=now),
            monthly_drawdown_max=monthly_dd_kill,
        )

        # R7.C1 + R8.C1: sticky-cooldown window. From the first trip, stay
        # tripped for `_kill_cooldown_days` calendar days regardless of fresh
        # telemetry, per the design doc's "Auto-resume after 7 calendar days
        # OR explicit manual reset."
        #
        # R8.C1 fix: after the sticky window elapses, the killswitch state
        # follows the real-time verdict but does NOT auto-rearm a new 7-day
        # window when conditions are still bad. Prior behavior cleared the
        # sticky timestamp at expiry, then set it again the same tick on a
        # persistently-bad metric — extending the window forever and hiding
        # the actual real-time state from the operator.
        in_sticky_window = (
            self._kill_first_tripped_at is not None
            and now < self._kill_first_tripped_at + timedelta(days=self._kill_cooldown_days)
        )
        if in_sticky_window:
            self.kill_switch_active = True
            if self.metrics_registry:
                reason = self._kill_reason or "cooldown"
                self.metrics_registry.set_kill_switch_active(reason)
                # R10.2: track every reason label set during the cycle.
                self._active_kill_reasons.add(reason)
            return

        was_active = self.kill_switch_active
        self.kill_switch_active = ks.tripped
        self._kill_reason = ks.reason
        if ks.tripped:
            # Open a new sticky window ONLY on a transition from clear → tripped.
            if not was_active:
                self._kill_first_tripped_at = now
                log.warning("killswitch_tripped", reason=ks.reason)
                await self._notify(
                    f"squeeze-hunter killswitch tripped: {ks.reason} "
                    f"(mode={self.mode}, at={now.isoformat()})"
                )
            if self.metrics_registry:
                reason = ks.reason or "unknown"
                self.metrics_registry.set_kill_switch_active(reason)
                # R10.2: remember every reason that was active in this cycle so
                # the eventual clear resets ALL of them. The trip reason can
                # transition mid-cycle (drawdown → data_stale) and Prometheus
                # tracks each labeled time series separately.
                self._active_kill_reasons.add(reason)
        elif self._kill_first_tripped_at is not None:
            # Conditions cleared — close the trip cycle so the next fresh trip
            # opens a new cooldown window.
            log.info("killswitch_cleared", prior_reason=self._kill_reason)
            # R9.4 + R10.2: reset every reason label that was active during the
            # cycle so Grafana shows all gauges back at 0.0. Resetting only the
            # most-recent reason left earlier reasons stuck at 1.0 forever.
            if self.metrics_registry is not None:
                for reason in self._active_kill_reasons:
                    self.metrics_registry.set_kill_switch_inactive(reason)
            self._active_kill_reasons.clear()
            self._kill_first_tripped_at = None

    def reset_killswitch(self: RuntimeContext) -> None:
        """R7.C1: explicit manual reset. Clears sticky cooldown state and
        un-trips the switch even if the cooldown window hasn't elapsed.

        The operator is responsible for verifying that the triggering
        condition has actually resolved before calling this.
        """
        log.warning(
            "killswitch_manual_reset",
            prior_reason=self._kill_reason,
            first_tripped_at=(
                self._kill_first_tripped_at.isoformat() if self._kill_first_tripped_at else None
            ),
        )
        # R9.4 + R10.2: reset every reason label set during the cycle so Grafana
        # reflects the manual reset. Resetting only `_kill_reason` would leave
        # any earlier transition-reason stuck at 1.0.
        if self.metrics_registry is not None:
            for reason in self._active_kill_reasons:
                self.metrics_registry.set_kill_switch_inactive(reason)
        self._active_kill_reasons.clear()
        self._kill_first_tripped_at = None
        self.kill_switch_active = False
        self._kill_reason = None

    async def tick_safe(self: RuntimeContext, now: datetime) -> bool:
        """Run tick() and swallow any exception so the scheduler keeps firing.

        Returns True on success, False if an exception was caught and logged.
        Use this for fire-and-forget scheduler callbacks where an unhandled
        exception in a Task would otherwise be silently dropped by asyncio.
        """
        try:
            await self.tick(now=now)
        except Exception:
            log.exception("tick_failed", as_of=now.isoformat())
            return False
        return True

    async def nightly_scan(self: RuntimeContext, now: datetime) -> None:
        """Nightly: scan the universe, refresh current_score for held positions,
        persist the ranked candidates for tomorrow's premarket_verify.

        Uses BacktestProvider in all modes — it reads historical parquet. In
        paper/live mode the cache must be kept current by a separate ingest
        job (Phase 4); otherwise the scan operates on stale data and f5
        (call OI velocity) will be 0 for every ticker. We log this limitation
        explicitly when in paper/live so it shows up in structured logs.

        R9.9 (Phase 4 entry-path checklist): Phase 3 publishes candidates to
        self.last_candidates but does NOT auto-enter positions. When the
        Phase 4 entry path is wired, the implementation MUST:
          1. Apply evaluate_gates with the SAME args as
             backtest/runner.py's call site (score_threshold,
             max_new_per_day, max_positions, position_cap, max_gross_exposure,
             plus a GateContext that suppresses on kill_switch_active). Any
             divergence violates the "same code path" design rule.
          2. Initialize ALL meta keys in lifecycle_state.positions[t]:
             entry_price, peak_price=entry_price, entry_score, current_score,
             bars_held=0, setup_type, qty, entry_commission. Missing any key
             will KeyError in lifecycle._process_one_position the next tick.
          3. Call self.telemetry.record_position(t, entry, entry) so
             worst_position_gap_pct sees the new position; otherwise the
             gap-through-stop killswitch arm is dead for that lot.
        """
        from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
        from squeeze_hunter.scan import run_scan

        if self.mode in {"paper", "live"}:
            log.warning(
                "nightly_scan_using_cache_only",
                mode=self.mode,
                note="f5 OI velocity will be 0 until a live options ingest job is added (Phase 4)",
            )

        # R5.C2: when the killswitch is active, suppress new-candidate
        # emission so the operator can't accidentally open a position that
        # the killswitch would have rejected (the entry-time gate from the
        # backtest runner has no equivalent in Phase 3's manual review
        # workflow). Existing positions still run through their stops in
        # tick() — that's the spec's "no panic-flatten" behavior.
        clock = Clock(now=now)
        provider = BacktestProvider(
            cache=self.cache,
            clock=clock,
            finra_publication_lag_bdays=self.settings.data.finra_publication_lag_bdays,
        )
        ranked = await run_scan(self.tickers, provider, now, self.settings)
        if self.kill_switch_active:
            # Round-12: the scan still runs so HELD positions get their
            # current_score refreshed below — the signal-decay stops read it,
            # and freezing it for the whole cooldown made them dead exactly
            # when a tripped portfolio needs them. Only candidate emission is
            # suppressed.
            log.warning(
                "nightly_scan_suppressed_killswitch_active",
                reason=self._kill_reason,
            )
            self.last_candidates = pd.DataFrame()
        else:
            self.last_candidates = ranked

        if not ranked.empty:
            ranked_by_ticker = ranked.set_index("ticker")["score"].to_dict()
            for ticker, meta in self.lifecycle_state.positions.items():
                if ticker in ranked_by_ticker:
                    meta["current_score"] = float(ranked_by_ticker[ticker])
                # If ticker isn't in scan output (e.g. dropped from universe),
                # keep prior current_score so the position isn't silently zeroed.
        log.info("nightly_scan_complete", n_candidates=len(ranked))

    async def nightly_scan_safe(self: RuntimeContext, now: datetime) -> bool:
        """Run nightly_scan() and swallow any exception so the scheduler keeps firing.

        Returns True on success, False if an exception was caught and logged.
        Mirrors the tick_safe contract.
        """
        try:
            await self.nightly_scan(now=now)
        except Exception:
            log.exception("nightly_scan_failed", as_of=now.isoformat())
            return False
        return True

    async def eod_close(self: RuntimeContext, now: datetime) -> None:
        """End-of-day: increment bars_held for every open position.

        This is the counter that powers the 21-trading-day time stop.
        Optionally snapshot daily P&L here in a future iteration (Phase 4).

        R4.5: skip increment on US federal holidays. The scheduler fires on
        every weekday, but NYSE is closed on ~9 federal holidays per year.
        Counting those as trading days would force the time-stop ~2 calendar
        weeks earlier than intended.
        """
        # Avoid a circular import: pull the holiday cache from the earnings
        # signal module which already maintains it.
        from squeeze_hunter.signals.earnings_reaction import _us_business_holidays

        et = now.astimezone(_NY)
        if et.date() in set(_us_business_holidays()):
            log.info("eod_close_skipped_holiday", date=et.date().isoformat())
            return

        for meta in self.lifecycle_state.positions.values():
            meta["bars_held"] = int(meta.get("bars_held", 0)) + 1
        log.info(
            "eod_close_complete",
            positions=len(self.lifecycle_state.positions),
        )

    async def eod_close_safe(self: RuntimeContext, now: datetime) -> bool:
        """Run eod_close() and swallow any exception so the scheduler keeps firing.

        Returns True on success, False if an exception was caught and logged.
        Mirrors the tick_safe contract.
        """
        try:
            await self.eod_close(now=now)
        except Exception:
            log.exception("eod_close_failed", as_of=now.isoformat())
            return False
        return True

    async def premarket_verify(self: RuntimeContext, now: datetime) -> None:
        """Premarket: re-check overnight news and the candidate list against fresh info.

        Phase 3 stub — logs intent so the cron actually fires and appears in
        structured logs. Phase 4 will add halt-list scraping, news scoring, and
        candidate-list filtering using last_candidates populated by nightly_scan.
        """
        log.info(
            "premarket_verify_stub",
            candidates_from_overnight=len(self.last_candidates)
            if self.last_candidates is not None
            else 0,
        )

    async def premarket_verify_safe(self: RuntimeContext, now: datetime) -> bool:
        """Run premarket_verify() and swallow any exception so the scheduler keeps firing.

        Returns True on success, False if an exception was caught and logged.
        Mirrors the tick_safe contract.
        """
        try:
            await self.premarket_verify(now=now)
        except Exception:
            log.exception("premarket_verify_failed", as_of=now.isoformat())
            return False
        return True

    async def _try_cleanup_partial_broker(self: RuntimeContext) -> None:
        """R4.4: best-effort disconnect when setup fails mid-way (e.g., timeout).

        Doesn't raise — we're already in an error path. Logs failures.
        """
        if self.broker is None:
            return
        try:
            if hasattr(self.broker, "disconnect"):
                await self.broker.disconnect()
        except (ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            # R8.Q-I2: narrow per CLAUDE.md. RuntimeError covers "event loop
            # closed" during shutdown. AttributeError / TypeError must still
            # propagate so a broker-impl bug surfaces.
            log.warning(
                "partial_broker_cleanup_failed",
                err=str(e),
                err_type=type(e).__name__,
            )
        finally:
            self.broker = None

    async def shutdown(self: RuntimeContext) -> None:
        if self.monitor_server is not None:
            await asyncio.to_thread(self.monitor_server.stop)
            self.monitor_server = None
        if self.broker is not None and hasattr(self.broker, "disconnect"):
            await self.broker.disconnect()
