"""Top-level runtime — wires settings, broker, scheduler, monitor into one process."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

import pandas as pd

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.base import IBroker
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.execution.lifecycle import LifecycleState, manage_positions
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.monitor.metrics import MetricsRegistry
from squeeze_hunter.risk.killswitch import evaluate_killswitch

if TYPE_CHECKING:
    from squeeze_hunter.risk.killswitch import KillSwitchInputs

log = get_logger("runtime")

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


@dataclass
class PortfolioTelemetry:
    """Tracks the inputs that feed evaluate_killswitch.

    All metrics are computed lazily on demand from recorded history.
    """

    equity_history: list[tuple[datetime, float]] = field(default_factory=list)
    position_marks: dict[str, tuple[float, float]] = field(default_factory=dict)
    # ticker -> (entry_price, mark_price); negative gap = adverse move
    last_broker_heartbeat: datetime | None = None
    data_freshness: dict[str, datetime] = field(default_factory=dict)
    critical_sources: set[str] = field(default_factory=lambda: {"ibkr_quotes"})

    def record_equity(self: PortfolioTelemetry, ts: datetime, equity_usd: float) -> None:
        self.equity_history.append((ts, equity_usd))

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
        recent = [(t, e) for t, e in self.equity_history if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        peak = max(e for _, e in recent)
        current = recent[-1][1]
        if peak <= 0:
            return 0.0
        return (current - peak) / peak

    def last_3_days_cumulative_pnl_pct(self: PortfolioTelemetry, as_of: datetime) -> float:
        cutoff = as_of - timedelta(days=3)
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
    telemetry: PortfolioTelemetry = field(default_factory=PortfolioTelemetry)
    # Populated by nightly_scan; read by premarket_verify the next morning.
    last_candidates: pd.DataFrame | None = None

    async def setup(self: RuntimeContext, connect_timeout_s: float = 30.0) -> None:
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
                # the process forever with no diagnostic. The supervisor can
                # then restart or alert.
                try:
                    await asyncio.wait_for(self.broker.connect(), timeout=connect_timeout_s)
                except TimeoutError:
                    log.error("paper_broker_connect_timeout", timeout_s=connect_timeout_s)
                    raise
            elif self.mode == "live":
                from squeeze_hunter.broker.ibkr import IBKRBroker

                self.broker = IBKRBroker(client_id=int(os.environ.get("IBKR_CLIENT_ID", "42")))
                try:
                    await asyncio.wait_for(self.broker.connect(), timeout=connect_timeout_s)
                except TimeoutError:
                    log.error("live_broker_connect_timeout", timeout_s=connect_timeout_s)
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
        except Exception:
            log.warning("broker_health_unreachable_at_setup")
            # Seed heartbeat anyway so disconnect timer measures from now.
            # If still down at next tick, broker_disconnected_for_seconds grows.
            self.telemetry.record_broker_heartbeat(datetime.now(UTC))

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
        if not _is_us_regular_session(now):
            log.debug("tick_skipped_outside_session", now=now.isoformat())
            return

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
        try:
            health = await self.broker.health()
            if health.connected:
                self.telemetry.record_broker_heartbeat(now)
        except Exception:
            log.warning("broker_health_check_failed")

        # Mark to market: fetch quotes, record position marks, collect prices.
        marks: dict[str, float] = {}
        for ticker, meta in self.lifecycle_state.positions.items():
            try:
                q = await self.broker.fetch_quote(ticker)
                price = q.last or q.bid or q.ask
                if price > 0:
                    marks[ticker] = price
                    self.telemetry.record_position(ticker, meta["entry_price"], price)
                    self.telemetry.record_data_freshness("ibkr_quotes", now)
            except Exception:
                pass

        # Update broker equity with current marks and record for telemetry.
        # Only append if there is no existing record already at this exact timestamp
        # (avoids overwriting pre-seeded telemetry in tests and backfill scenarios).
        if isinstance(self.broker, SimulatorBroker):
            self.broker.mark_to_market(marks, now)
            ts_set = {t for t, _ in self.telemetry.equity_history}
            if now not in ts_set:
                self.telemetry.record_equity(now, self.broker.equity)

        # Build killswitch inputs from real telemetry and evaluate.
        ks = evaluate_killswitch(self.telemetry.to_killswitch_inputs(as_of=now))
        self.kill_switch_active = ks.tripped
        self._kill_reason = ks.reason
        if ks.tripped:
            log.warning("killswitch_tripped", reason=ks.reason)
            if self.metrics_registry:
                self.metrics_registry.set_kill_switch_active(ks.reason or "unknown")

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
        """
        from squeeze_hunter.data.providers.backtest import BacktestProvider, Clock
        from squeeze_hunter.scan import run_scan

        if self.mode in {"paper", "live"}:
            log.warning(
                "nightly_scan_using_cache_only",
                mode=self.mode,
                note="f5 OI velocity will be 0 until a live options ingest job is added (Phase 4)",
            )

        clock = Clock(now=now)
        provider = BacktestProvider(cache=self.cache, clock=clock)
        ranked = await run_scan(self.tickers, provider, now, self.settings)
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
        """
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

    async def shutdown(self: RuntimeContext) -> None:
        if self.broker is not None and hasattr(self.broker, "disconnect"):
            await self.broker.disconnect()
