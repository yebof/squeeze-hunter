"""DataProvider Protocol — the only contract data sources must satisfy."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Protocol, runtime_checkable

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    Quote,
    RedditMention,
    ShortInterest,
)

Resolution = Literal["1m", "5m", "1h", "1d"]


@runtime_checkable
class DataProvider(Protocol):
    name: str
    capabilities: frozenset[str]  # subset of {"bars","quote","options","si","earnings","sentiment"}

    async def fetch_bars(
        self: DataProvider,
        ticker: str,
        start: datetime,
        end: datetime,
        resolution: Resolution = "1d",
    ) -> list[Bar]: ...

    async def fetch_quote(self: DataProvider, ticker: str) -> Quote: ...

    async def fetch_option_chain(
        self: DataProvider, ticker: str, expiry: date | None = None
    ) -> OptionChain: ...

    async def fetch_short_interest(
        self: DataProvider, ticker: str, since: date | None = None
    ) -> list[ShortInterest]: ...

    async def fetch_earnings(
        self: DataProvider, ticker: str, since: date | None = None
    ) -> list[EarningsEvent]: ...

    async def fetch_sentiment(
        self: DataProvider, ticker: str, as_of: datetime
    ) -> RedditMention | None: ...
