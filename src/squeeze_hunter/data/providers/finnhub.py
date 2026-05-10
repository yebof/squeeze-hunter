"""FinnhubProvider — free-tier earnings calendar."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import finnhub

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    Quote,
    RedditMention,
    ShortInterest,
)


@dataclass
class FinnhubProvider:
    api_key: str
    name: str = "finnhub"
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({"earnings"}))

    def __post_init__(self: FinnhubProvider) -> None:
        self._client = finnhub.Client(api_key=self.api_key) if self.api_key else None

    def _call_calendar(self: FinnhubProvider, ticker: str, frm: str, to: str) -> dict[str, Any]:
        if self._client is None:
            return {"earningsCalendar": []}
        return self._client.earnings_calendar(_from=frm, to=to, symbol=ticker, international=False)

    async def fetch_earnings(
        self: FinnhubProvider, ticker: str, since: date | None = None
    ) -> list[EarningsEvent]:
        frm = (since or (date.today() - timedelta(days=365 * 2))).isoformat()
        to = (date.today() + timedelta(days=180)).isoformat()
        raw = await asyncio.to_thread(self._call_calendar, ticker, frm, to)
        out: list[EarningsEvent] = []
        for ev in raw.get("earningsCalendar", []):
            d = date.fromisoformat(ev["date"])
            hour = (ev.get("hour") or "").lower()
            ts = datetime(d.year, d.month, d.day, 20 if hour == "amc" else 12, 30, tzinfo=UTC)
            out.append(
                EarningsEvent(
                    ticker=ticker,
                    report_at=ts,
                    actual_eps=ev.get("epsActual"),
                    estimate_eps=ev.get("epsEstimate"),
                    actual_revenue=ev.get("revenueActual"),
                    estimate_revenue=ev.get("revenueEstimate"),
                )
            )
        return out

    # -- protocol stubs --
    async def fetch_bars(self: FinnhubProvider, *_a: object, **_kw: object) -> list[Bar]:
        return []

    async def fetch_quote(self: FinnhubProvider, ticker: str) -> Quote:
        raise NotImplementedError

    async def fetch_option_chain(self: FinnhubProvider, *_a: object, **_kw: object) -> OptionChain:
        return OptionChain(underlying="", as_of=datetime.now(UTC), spot=0.0, quotes=[])

    async def fetch_short_interest(
        self: FinnhubProvider, *_a: object, **_kw: object
    ) -> list[ShortInterest]:
        return []

    async def fetch_sentiment(
        self: FinnhubProvider, *_a: object, **_kw: object
    ) -> RedditMention | None:
        return None
