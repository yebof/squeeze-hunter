"""Top-level runtime — wires settings, broker, scheduler, monitor into one process."""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pandas as pd

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.base import IBroker
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.execution.decisions import EntryDecision
from squeeze_hunter.execution.entries import EntryPath
from squeeze_hunter.execution.lifecycle import LifecycleState, manage_positions
from squeeze_hunter.execution.orders import OrderTracker
from squeeze_hunter.execution.reconcile import reconcile_book
from squeeze_hunter.ingest.eod import ingest_eod
from squeeze_hunter.ingest.freshness import dataset_age_days
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.monitor.alerts import AlertSender, Severity
from squeeze_hunter.monitor.http import MonitorServer, start_monitor_server
from squeeze_hunter.monitor.metrics import MetricsRegistry
from squeeze_hunter.risk.killswitch_controller import KillswitchController
from squeeze_hunter.store.state import JsonStateStore, StateStore
from squeeze_hunter.telemetry import PortfolioTelemetry
from squeeze_hunter.trading_calendar import (
    NY,
    is_regular_session,
    is_trading_day,
    session_open_utc,
)

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


# R3.2 / P5: the session window lives in trading_calendar (single source of
# truth for live and backtest); the old private names stay as aliases.
_is_us_regular_session = is_regular_session
_session_open_utc = session_open_utc


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
    # P4: the killswitch state machine and its gauge bookkeeping live in
    # one controller; the facade properties below keep the old names.
    killswitch: KillswitchController = field(default_factory=KillswitchController)
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
    # P4: the live entry path (plan / execute / settle) owns planned and
    # pending entries; built in __post_init__ because it needs the book.
    entries: EntryPath = field(init=False, repr=False)
    # P2: snapshot store (None = no persistence). Built from data.state_path in
    # setup() unless injected.
    state_store: StateStore | None = None

    def __post_init__(self: RuntimeContext) -> None:
        self.entries = EntryPath(
            cache=self.cache,
            settings=self.settings,
            tickers=self.tickers,
            book=self.lifecycle_state,
            telemetry=self.telemetry,
            mode=self.mode,
        )

    # ---- facade properties: the CLI, the monitor and the tests read these names
    @property
    def kill_switch_active(self: RuntimeContext) -> bool:
        return self.killswitch.active

    @kill_switch_active.setter
    def kill_switch_active(self: RuntimeContext, value: bool) -> None:
        self.killswitch.active = bool(value)

    @property
    def _kill_reason(self: RuntimeContext) -> str | None:
        return self.killswitch.reason

    @_kill_reason.setter
    def _kill_reason(self: RuntimeContext, value: str | None) -> None:
        self.killswitch.reason = value

    @property
    def _kill_first_tripped_at(self: RuntimeContext) -> datetime | None:
        return self.killswitch.first_tripped_at

    @_kill_first_tripped_at.setter
    def _kill_first_tripped_at(self: RuntimeContext, value: datetime | None) -> None:
        self.killswitch.first_tripped_at = value

    @property
    def _active_kill_reasons(self: RuntimeContext) -> set[str]:
        return self.killswitch.active_reasons

    @property
    def _kill_cooldown_days(self: RuntimeContext) -> int:
        return self.killswitch.cooldown_days

    @property
    def planned_entries(self: RuntimeContext) -> list[EntryDecision]:
        return self.entries.planned

    @planned_entries.setter
    def planned_entries(self: RuntimeContext, value: list[EntryDecision]) -> None:
        self.entries.planned = list(value)

    @property
    def pending_buys(self: RuntimeContext) -> OrderTracker:
        return self.entries.pending

    @pending_buys.setter
    def pending_buys(self: RuntimeContext, value: OrderTracker) -> None:
        self.entries.pending = value

    async def setup(
        self: RuntimeContext, connect_timeout_s: float = 30.0, *, now: datetime | None = None
    ) -> None:
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
        # P9: the sticky cooldown is a YAML knob, mirrored by the backtest runner.
        self.killswitch.cooldown_days = self.settings.risk.killswitch.cooldown_days
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
                from squeeze_hunter.broker.ibkr import IBKRBroker, require_live_port

                # Round-13: refuse the paper-port default in live mode.
                require_live_port()
                self.broker = IBKRBroker(client_id=int(os.environ.get("IBKR_CLIENT_ID", "42")))
                try:
                    await asyncio.wait_for(self.broker.connect(), timeout=connect_timeout_s)
                except TimeoutError:
                    log.error("live_broker_connect_timeout", timeout_s=connect_timeout_s)
                    await self._try_cleanup_partial_broker()
                    raise
            else:
                raise ValueError(f"unknown mode: {self.mode}")
        self.entries.broker = self.broker
        self.metrics_registry = MetricsRegistry()
        # R7: seed the broker heartbeat at setup so broker_disconnected_for_seconds
        # measures elapsed time correctly. Without this seed, a startup where the
        # broker is unreachable shows 0 disconnect-seconds forever (the killswitch
        # never trips because last_broker_heartbeat stays None).
        try:
            health = await self.broker.health()
            if health.connected:
                self.telemetry.record_broker_heartbeat(datetime.now(UTC))
                # Round-13: seed quote freshness too, so the data_stale arm
                # measures from startup instead of reading 0 ("no sources")
                # until the first fresh quote ever arrives.
                self.telemetry.record_data_freshness("ibkr_quotes", datetime.now(UTC))
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
            self.telemetry.record_data_freshness("ibkr_quotes", datetime.now(UTC))
        # Round-12: alert channel (AlertSender had no caller — killswitch trips
        # only logged) and the /metrics + /health endpoint (Prometheus scraped
        # an empty port). Both were promised by spec §7.
        if self.alerts is None:  # an injected sender (tests, embedding) wins
            self.alerts = _alert_sender_from_env()
        if self.settings.monitor.http_port > 0 and self.monitor_server is None:
            self.monitor_server = start_monitor_server(
                self,
                port=self.settings.monitor.http_port,
                host=self.settings.monitor.http_host,
            )
        # P2: resume from the last snapshot, then let the broker correct it.
        if self.state_store is None and self.settings.data.state_path:
            self.state_store = JsonStateStore(Path(self.settings.data.state_path))
        if self.state_store is not None:
            snapshot = self.state_store.load()
            if snapshot:
                self._restore(snapshot)
        # P3: a buy that filled while we were down becomes a position with its
        # real meta BEFORE reconciliation could adopt it as an unknown lot.
        startup_now = now or datetime.now(UTC)
        await self.entries.settle_pending(startup_now)
        await self._reconcile_with_broker(startup_now, full=True, source="startup")
        self._persist()

    async def _notify(
        self: RuntimeContext, text: str, *, severity: Severity = Severity.HIGH
    ) -> None:
        """Push an alert (HIGH by default); delivery failure must never break a tick."""
        if self.alerts is None:
            log.warning("alert_channel_not_configured", text=text)
            return
        try:
            await self.alerts.send(text, severity=severity)
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
        await self.entries.settle_pending(now)
        await self.entries.execute_planned(now, kill_active=self.kill_switch_active)
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
        # Round-13: a healthy socket refreshes quote freshness only when there
        # is nothing to quote (flat portfolio). With positions, freshness comes
        # from an actually-delivered quote in the mark loop below —
        # IB.isConnected() is pure socket state, so the data_stale arm could
        # never trip while the socket was up even with frozen market data.
        if broker_healthy and not self.lifecycle_state.positions:
            self.telemetry.record_data_freshness("ibkr_quotes", now)
        self.last_broker_healthy = broker_healthy
        # P2: 60 s reconcile — adopt the broker's quantities; positions with an
        # exit in flight are left to the daemon's own reconcile.
        if broker_healthy:
            await self._reconcile_with_broker(now, full=False, source="tick")

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
        # Round-13: these two gauges do not depend on NAV; nesting them under
        # the equity check left sh_broker_connected at 0 on a healthy broker
        # whenever NetLiquidation was unavailable.
        if self.metrics_registry is not None:
            self.metrics_registry.position_count.set(len(self.lifecycle_state.positions))
            self.metrics_registry.broker_connected.set(1.0 if broker_healthy else 0.0)

        # P4: verdict + sticky cooldown + gauges live in the controller; this
        # method only logs and alerts on the transition.
        prior_reason = self._kill_reason
        transition = self.killswitch.evaluate(
            self.telemetry.to_killswitch_inputs(as_of=now),
            self.settings,
            now,
            metrics=self.metrics_registry,
        )
        if transition == "tripped":
            log.warning("killswitch_tripped", reason=self._kill_reason)
            await self._notify(
                f"squeeze-hunter killswitch tripped: {self._kill_reason} "
                f"(mode={self.mode}, at={now.isoformat()})"
            )
        elif transition == "cleared":
            log.info("killswitch_cleared", prior_reason=prior_reason)
        self._persist()

    def _data_freshness_problems(
        self: RuntimeContext, now: datetime, *, critical_only: bool = False
    ) -> list[str]:
        """Datasets whose newest point is older than its budget (or unknown).
        `critical_only` restricts to data.critical_datasets (the entry gate)."""
        budgets = {
            "bars": self.settings.data.bars_max_age_days,
            "short_interest": self.settings.data.short_interest_max_age_days,
            "earnings": self.settings.data.earnings_max_age_days,
        }
        critical = set(self.settings.data.critical_datasets)
        problems: list[str] = []
        for dataset, budget in budgets.items():
            if critical_only and dataset not in critical:
                continue
            age = dataset_age_days(self.cache.root, dataset, now)
            if age is None:
                problems.append(f"{dataset}: never ingested (max {budget:g}d)")
            elif age > budget:
                problems.append(f"{dataset}: {age:.1f}d old (max {budget:g}d)")
        return problems

    async def ingest_eod(self: RuntimeContext, now: datetime) -> None:
        """17:00 ET: bring bars / short interest / earnings up to date."""
        report = await ingest_eod(self.tickers, self.cache, self.settings, now)
        if not report.ok:
            await self._notify(
                "squeeze-hunter EOD ingest problems: "
                f"bars_failed={report.bars_failed} finra={report.finra} "
                f"earnings={report.earnings}",
                severity=Severity.LOW,
            )
        self._persist()

    async def ingest_eod_safe(self: RuntimeContext, now: datetime) -> bool:
        try:
            await self.ingest_eod(now=now)
        except Exception:
            log.exception("ingest_eod_failed", as_of=now.isoformat())
            return False
        return True

    # ------------------------------------------------------------------ P2
    def _snapshot(self: RuntimeContext) -> dict[str, Any]:
        return {
            "version": 1,
            "mode": self.mode,
            "positions": self.lifecycle_state.positions,
            "planned_entries": [asdict(d) for d in self.planned_entries],
            "pending_buys": self.pending_buys.to_snapshot(),
            "killswitch": self.killswitch.to_snapshot(),
            "telemetry": {
                "equity_history": [
                    [ts.isoformat(), eq] for ts, eq in self.telemetry.equity_history
                ],
                "equity_peak_per_day": {
                    d.isoformat(): peak for d, peak in self.telemetry.equity_peak_per_day.items()
                },
            },
        }

    def _persist(self: RuntimeContext) -> None:
        if self.state_store is None:
            return
        try:
            self.state_store.save(self._snapshot())
        except OSError as e:
            # Persistence must never take the trading loop down; the next job
            # retries. Programming errors (TypeError from an unserialisable
            # value) propagate so they surface in tick_safe's logs.
            log.error("state_persist_failed", err=str(e), err_type=type(e).__name__)

    def _restore(self: RuntimeContext, snap: dict[str, Any]) -> None:
        positions = snap.get("positions") or {}
        self.lifecycle_state.positions = {
            str(t): dict(meta) for t, meta in positions.items() if isinstance(meta, dict)
        }
        for t, meta in self.lifecycle_state.positions.items():
            self.telemetry.record_position(
                t, float(meta["entry_price"]), float(meta["entry_price"])
            )
        self.planned_entries = [
            EntryDecision(**d) for d in (snap.get("planned_entries") or []) if isinstance(d, dict)
        ]
        self.pending_buys = OrderTracker.from_snapshot(snap.get("pending_buys"))
        self.killswitch.restore(snap.get("killswitch") or {})
        tele = snap.get("telemetry") or {}
        self.telemetry.equity_history = [
            (datetime.fromisoformat(ts), float(eq)) for ts, eq in tele.get("equity_history") or []
        ]
        self.telemetry.equity_peak_per_day = {
            date.fromisoformat(d): float(peak)
            for d, peak in (tele.get("equity_peak_per_day") or {}).items()
        }
        log.info(
            "state_restored",
            positions=len(self.lifecycle_state.positions),
            planned_entries=len(self.planned_entries),
            kill_switch_active=self.kill_switch_active,
            saved_at=snap.get("saved_at"),
        )

    async def _reconcile_with_broker(
        self: RuntimeContext, now: datetime, *, full: bool, source: str
    ) -> None:
        """P2: make the local book agree with the broker.

        - A local position the broker does not hold is a phantom: dropped.
        - A quantity mismatch adopts the broker's quantity.
        - A broker holding we do not know is adopted with conservative meta
          (entry = avg cost, setup Mixed, score 0 so signal-decay never fires,
          bars_held 0) so the hard / trailing / time stops manage it.
        Positions with an exit in flight are skipped on the 60 s pass: the
        daemon's own pending-exit reconcile owns them. `full` (startup / EOD)
        pushes an alert on any drift; the tick pass only logs.
        """
        if self.broker is None:
            return
        if isinstance(self.broker, SimulatorBroker) and self.state_store is None:
            # Pure in-memory sim (tests, ad-hoc harnesses) seeds the book
            # directly; there is no external truth to reconcile against.
            # Paper / live always reconcile; sim does once persistence is on.
            return
        try:
            drift = await reconcile_book(
                self.broker,
                self.lifecycle_state,
                self.telemetry,
                full=full,
                # P3: a holding that a pending buy is about to explain is not "unknown".
                skip_tickers={r.ticker for r in self.pending_buys.open()},
            )
        except _TRANSIENT_IO_ERRORS as e:
            log.warning("reconcile_positions_unavailable", source=source, err=str(e))
            return
        if not drift:
            return
        log.warning("reconcile_drift", source=source, items=drift)
        if full:
            await self._notify(
                f"squeeze-hunter reconciliation ({source}, mode={self.mode}): " + "; ".join(drift)
            )

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
        self.killswitch.reset(metrics=self.metrics_registry)

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
        stale = self._data_freshness_problems(now)
        if stale:
            log.warning("nightly_scan_on_stale_data", problems=stale)
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
        self._persist()

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
        et = now.astimezone(NY)
        if not is_trading_day(et.date()):
            log.info("eod_close_skipped_holiday", date=et.date().isoformat())
            return

        # P2: full EOD reconciliation; any drift alerts the operator.
        await self._reconcile_with_broker(now, full=True, source="eod")
        for meta in self.lifecycle_state.positions.values():
            meta["bars_held"] = int(meta.get("bars_held", 0)) + 1
        log.info(
            "eod_close_complete",
            positions=len(self.lifecycle_state.positions),
        )
        self._persist()

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
        n_candidates = len(self.last_candidates) if self.last_candidates is not None else 0
        log.info("premarket_verify", candidates_from_overnight=n_candidates)
        self.planned_entries = []
        if not self.settings.execution.auto_enter:
            return
        # P6: never size entries off a stale cache.
        problems = self._data_freshness_problems(now, critical_only=True)
        if problems and self.settings.data.require_fresh_for_entries:
            log.warning("premarket_entries_refused_stale_data", problems=problems)
            await self._notify(
                "squeeze-hunter: automatic entries refused, data stale — " + "; ".join(problems)
            )
            self._persist()
            return
        await self.entries.plan(
            now,
            self.last_candidates,
            kill_active=self.kill_switch_active,
            kill_reason=self._kill_reason,
        )
        self._persist()

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
        self._persist()
        if self.monitor_server is not None:
            await asyncio.to_thread(self.monitor_server.stop)
            self.monitor_server = None
        if self.broker is not None and hasattr(self.broker, "disconnect"):
            await self.broker.disconnect()
