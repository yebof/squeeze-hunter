"""Top-level runtime — wires settings, broker, scheduler, monitor into one process."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

from squeeze_hunter.backtest.cost_model import StockCostModel
from squeeze_hunter.broker.base import IBroker
from squeeze_hunter.broker.simulator import SimulatorBroker
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.execution.lifecycle import LifecycleState, manage_positions
from squeeze_hunter.logging_setup import get_logger
from squeeze_hunter.monitor.metrics import MetricsRegistry
from squeeze_hunter.risk.killswitch import KillSwitchInputs, evaluate_killswitch

log = get_logger("runtime")


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

        # Telemetry for killswitch (TODO: wire real numbers in Phase 4)
        ks = evaluate_killswitch(
            KillSwitchInputs(
                as_of=now,
                rolling_30d_max_drawdown=0.0,
                last_3_days_cumulative_pnl_pct=0.0,
                worst_position_gap_pct=0.0,
                broker_disconnected_for_seconds=0,
                critical_data_stale_for_seconds=0,
            )
        )
        self.kill_switch_active = ks.tripped
        self._kill_reason = ks.reason

    async def shutdown(self: RuntimeContext) -> None:
        if self.broker is not None and hasattr(self.broker, "disconnect"):
            await self.broker.disconnect()
