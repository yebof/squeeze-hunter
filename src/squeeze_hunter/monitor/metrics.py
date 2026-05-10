"""Prometheus metrics — single registry per process."""

from __future__ import annotations

from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest


@dataclass
class MetricsRegistry:
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self: MetricsRegistry) -> None:
        self.signals_computed = Counter(
            "sh_signals_computed_total",
            "Signals computed",
            ["factor"],
            registry=self.registry,
        )
        self.candidates = Counter(
            "sh_candidates_total",
            "Ranked candidates emitted",
            ["setup_type"],
            registry=self.registry,
        )
        self.orders_submitted = Counter(
            "sh_orders_submitted_total",
            "Orders submitted",
            ["side", "status"],
            registry=self.registry,
        )
        self.orders_filled = Counter(
            "sh_orders_filled_total",
            "Orders filled",
            ["side"],
            registry=self.registry,
        )
        self.position_count = Gauge(
            "sh_position_count",
            "Open positions",
            registry=self.registry,
        )
        self.gross_exposure = Gauge(
            "sh_gross_exposure_pct",
            "Gross exposure as fraction of equity",
            registry=self.registry,
        )
        self.equity = Gauge(
            "sh_equity_usd",
            "Total equity in USD",
            registry=self.registry,
        )
        self.daily_pnl = Gauge(
            "sh_daily_pnl_usd",
            "Realized + unrealized daily P&L",
            registry=self.registry,
        )
        self.drawdown = Gauge(
            "sh_drawdown_pct",
            "Current drawdown from rolling-30d peak",
            registry=self.registry,
        )
        self.kill_switch = Gauge(
            "sh_kill_switch_active",
            "Killswitch status (1=active)",
            ["reason"],
            registry=self.registry,
        )
        self.broker_connected = Gauge(
            "sh_broker_connected",
            "Broker connected (1=yes)",
            registry=self.registry,
        )
        self.db_connected = Gauge(
            "sh_db_connected",
            "Postgres connected (1=yes)",
            registry=self.registry,
        )

    def record_order_submitted(self: MetricsRegistry, side: str, status: str) -> None:
        self.orders_submitted.labels(side=side, status=status).inc()

    def record_order_filled(self: MetricsRegistry, side: str) -> None:
        self.orders_filled.labels(side=side).inc()

    def set_equity(self: MetricsRegistry, usd: float) -> None:
        self.equity.set(usd)

    def set_kill_switch_active(self: MetricsRegistry, reason: str) -> None:
        self.kill_switch.labels(reason=reason).set(1.0)

    def render(self: MetricsRegistry) -> str:
        return generate_latest(self.registry).decode("utf-8")
