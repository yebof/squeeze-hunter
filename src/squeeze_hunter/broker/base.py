"""IBroker Protocol — the only contract live/paper/sim brokers must satisfy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(slots=True, frozen=True)
class Quote:
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
