"""YahooProvider — yfinance wrapper. Used for EOD history and as fallback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal, cast

import pandas as pd
import yfinance as yf

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    OptionQuote,
    RedditMention,
    ShortInterest,
)


def _yf_history(ticker: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    """Synchronous yfinance call — kept top-level so tests can patch it."""
    return yf.Ticker(ticker).history(start=start, end=end, interval=interval, auto_adjust=False)


@dataclass
class YahooProvider:
    name: str = "yahoo"
    capabilities: frozenset[str] = frozenset({"bars", "options", "earnings"})

    async def fetch_bars(
        self: YahooProvider,
        ticker: str,
        start: datetime,
        end: datetime,
        resolution: str = "1d",
    ) -> list[Bar]:
        interval = {"1d": "1d", "1h": "1h", "5m": "5m", "1m": "1m"}.get(resolution, "1d")
        df = await asyncio.to_thread(_yf_history, ticker, start, end, interval)
        if df.empty:
            return []
        df.index = pd.to_datetime(df.index, utc=True)
        return [
            Bar(
                ticker=ticker,
                ts=cast(pd.Timestamp, ts).to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
            for ts, row in df.iterrows()
        ]

    async def fetch_quote(self: YahooProvider, ticker: str) -> None:  # type: ignore[override]
        raise NotImplementedError("YahooProvider does not provide real-time quotes")

    async def fetch_option_chain(
        self: YahooProvider, ticker: str, expiry: date | None = None
    ) -> OptionChain:
        def _go() -> OptionChain:
            t = yf.Ticker(ticker)
            spot = float(t.history(period="1d")["Close"].iloc[-1])
            expiries = t.options
            target = expiry.isoformat() if expiry else (expiries[0] if expiries else None)
            quotes: list[OptionQuote] = []
            if target:
                chain = t.option_chain(target)
                for df, right in [
                    (chain.calls, "C"),
                    (chain.puts, "P"),
                ]:
                    r = cast(Literal["C", "P"], right)
                    for _, row in df.iterrows():
                        quotes.append(
                            OptionQuote(
                                strike=float(row["strike"]),
                                expiry=date.fromisoformat(target),
                                right=r,
                                open_interest=int(row.get("openInterest") or 0),
                                volume=int(row.get("volume") or 0),
                                implied_vol=float(row.get("impliedVolatility") or 0.0),
                                bid=float(row.get("bid") or 0.0),
                                ask=float(row.get("ask") or 0.0),
                            )
                        )
            return OptionChain(
                underlying=ticker,
                as_of=datetime.now(UTC),
                spot=spot,
                quotes=quotes,
            )

        return await asyncio.to_thread(_go)

    async def fetch_short_interest(
        self: YahooProvider, ticker: str, since: date | None = None
    ) -> list[ShortInterest]:
        return []

    async def fetch_earnings(
        self: YahooProvider, ticker: str, since: date | None = None
    ) -> list[EarningsEvent]:
        def _go() -> list[EarningsEvent]:
            t = yf.Ticker(ticker)
            cal = t.calendar
            events: list[EarningsEvent] = []
            if isinstance(cal, dict) and cal.get("Earnings Date"):
                raw_ts = pd.Timestamp(cal["Earnings Date"][0]).tz_localize("UTC")
                if raw_ts is not pd.NaT:
                    ts: datetime = cast(datetime, raw_ts.to_pydatetime())
                    events.append(
                        EarningsEvent(
                            ticker=ticker,
                            report_at=ts,
                            estimate_eps=cal.get("Earnings Average"),
                        )
                    )
            return events

        return await asyncio.to_thread(_go)

    async def get_float_shares(self: YahooProvider, ticker: str) -> int | None:
        """Return current float shares (sharesOutstanding as fallback). None if unavailable.

        Note: Yahoo only exposes current float, not historical. When this value is stored
        alongside historical SI records the float is time-leaked (today's float applied to
        past SI dates). This is an accepted pragmatic simplification — reconstructing
        historical floats from 10-Q filings is out of scope.
        """

        def _go() -> int | None:
            info = yf.Ticker(ticker).info or {}
            f = info.get("floatShares") or info.get("sharesOutstanding")
            return int(f) if f else None

        return await asyncio.to_thread(_go)

    async def fetch_sentiment(
        self: YahooProvider, ticker: str, as_of: datetime
    ) -> RedditMention | None:
        return None
