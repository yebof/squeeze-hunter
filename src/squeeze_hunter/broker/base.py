"""IBroker Protocol — the only contract live/paper/sim brokers must satisfy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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


class IBroker(Protocol):
    name: str

    async def connect(self: IBroker) -> None: ...
    async def disconnect(self: IBroker) -> None: ...
    async def fetch_quote(self: IBroker, ticker: str) -> Quote: ...
    async def health(self: IBroker) -> BrokerHealth: ...
