"""IBroker Protocol — the only contract live/paper/sim brokers must satisfy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(slots=True, frozen=True)
class Quote:
    """Broker-side quote returned by IBroker.fetch_quote.

    R6.M6 note: there is a second `Quote` model in
    `squeeze_hunter.data.schema` used by DataProvider.fetch_quote. The two
    intentionally diverge:
      - broker.base.Quote (this one): ticker/bid/ask/last/timestamp_ns —
        nanosecond epoch timestamp, no sizes. Optimized for execution-loop
        round-trips.
      - data.schema.Quote: ticker/ts/bid/ask/last/bid_size/ask_size —
        datetime + sizes, used by historical / analytical paths.

    Callers in lifecycle.py and runtime.py access only the common fields
    (bid/ask/last/ticker). Do NOT add a field to one without the other; if
    you need a common attribute, add it to BOTH. A future refactor could
    unify them, but doing so would touch every provider implementation.
    """

    ticker: str
    bid: float
    ask: float
    last: float
    timestamp_ns: int


@dataclass(slots=True, frozen=True)
class BrokerHealth:
    connected: bool
    last_ping_ms: int
    account: str


@dataclass(slots=True, frozen=True)
class BrokerOrder:
    broker_order_id: str
    ticker: str
    side: str  # "buy" | "sell"
    qty: int
    limit_price: float | None
    status: str  # "pending" | "filled" | "partial" | "cancelled" | "rejected"
    filled_qty: int = 0
    avg_fill_price: float | None = None


@runtime_checkable
class IBroker(Protocol):
    name: str

    async def connect(self: IBroker) -> None: ...
    async def disconnect(self: IBroker) -> None: ...
    async def fetch_quote(self: IBroker, ticker: str) -> Quote: ...
    async def health(self: IBroker) -> BrokerHealth: ...

    async def submit_buy(
        self: IBroker,
        ticker: str,
        qty: int,
        limit_price: float | None,
        ts: datetime,
    ) -> BrokerOrder: ...

    async def submit_sell(
        self: IBroker,
        ticker: str,
        qty: int,
        limit_price: float | None,
        ts: datetime,
    ) -> BrokerOrder: ...

    async def cancel_order(self: IBroker, broker_order_id: str) -> bool: ...

    async def get_open_orders(self: IBroker) -> list[BrokerOrder]: ...

    async def get_equity_usd(self: IBroker) -> float | None:
        """Return total account equity (NAV) in USD, or None if unavailable.

        R4.1: this powers PortfolioTelemetry.equity_history in paper/live mode.
        Without it, rolling_30d_max_drawdown and last_3_days_cumulative_pnl_pct
        always return 0 outside sim mode → the drawdown-based killswitch arms
        are dead in production.
        """
        ...
