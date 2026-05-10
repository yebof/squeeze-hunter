"""Top-level runtime — wires settings, broker, scheduler, monitor into one process."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast

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

    async def setup(self: RuntimeContext) -> None:
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
                await self.broker.connect()
            elif self.mode == "live":
                from squeeze_hunter.broker.ibkr import IBKRBroker

                self.broker = IBKRBroker(client_id=int(os.environ.get("IBKR_CLIENT_ID", "42")))
                await self.broker.connect()
            else:
                raise ValueError(f"unknown mode: {self.mode}")
        self.metrics_registry = MetricsRegistry()

    async def tick(self: RuntimeContext, now: datetime) -> None:
        """One intraday tick: manage positions + check killswitch."""
        if self.broker is None or self.metrics_registry is None:
            raise RuntimeError("setup() not called")
        await manage_positions(self.lifecycle_state, self.broker, now)

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

    async def shutdown(self: RuntimeContext) -> None:
        if self.broker is not None and hasattr(self.broker, "disconnect"):
            await self.broker.disconnect()
