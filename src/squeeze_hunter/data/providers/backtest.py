"""BacktestProvider — replays cached parquet history with a clock that
prevents lookahead bias by construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    Quote,
    RedditMention,
    ShortInterest,
)


@dataclass
class Clock:
    now: datetime

    def advance_to(self: Clock, t: datetime) -> None:
        if t < self.now:
            raise ValueError("clock cannot go backwards")
        self.now = t


@dataclass
class BacktestProvider:
    cache: ParquetCache
    clock: Clock
    name: str = "backtest"
    capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset({"bars", "options", "si", "earnings", "sentiment"})
    )

    async def fetch_bars(
        self: BacktestProvider,
        ticker: str,
        start: datetime,
        end: datetime,
        resolution: str = "1d",
    ) -> list[Bar]:
        df = self.cache.read_partition("bars", ticker)
        if df.empty:
            return []
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        mask = (df["ts"] >= start) & (df["ts"] <= end) & (df["ts"] <= self.clock.now)
        return [
            Bar(
                ticker=row["ticker"],
                ts=row["ts"].to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )
            for _, row in df[mask].iterrows()
        ]

    async def fetch_quote(self: BacktestProvider, ticker: str) -> Quote:
        bars = await self.fetch_bars(
            ticker,
            self.clock.now.replace(hour=0, minute=0, second=0, microsecond=0),
            self.clock.now,
        )
        if not bars:
            raise LookupError(f"no bars for {ticker} as of {self.clock.now}")
        b = bars[-1]
        return Quote(ticker=ticker, ts=b.ts, bid=b.close, ask=b.close, last=b.close)

    async def fetch_option_chain(
        self: BacktestProvider, ticker: str, expiry: date | None = None
    ) -> OptionChain:
        df = self.cache.read_partition("options", f"{ticker}__{self.clock.now.date().isoformat()}")
        if df.empty:
            return OptionChain(underlying=ticker, as_of=self.clock.now, spot=0.0, quotes=[])
        from squeeze_hunter.data.schema import OptionQuote

        quotes = [
            OptionQuote(
                strike=float(r["strike"]),
                expiry=date.fromisoformat(r["expiry"]),
                right=r["right"],
                open_interest=int(r["open_interest"]),
                volume=int(r["volume"]),
                implied_vol=float(r["implied_vol"]),
                bid=float(r.get("bid", 0.0)),
                ask=float(r.get("ask", 0.0)),
            )
            for _, r in df.iterrows()
        ]
        spot = float(df["spot"].iloc[0]) if "spot" in df.columns else 0.0
        return OptionChain(underlying=ticker, as_of=self.clock.now, spot=spot, quotes=quotes)

    async def fetch_short_interest(
        self: BacktestProvider, ticker: str, since: date | None = None
    ) -> list[ShortInterest]:
        df = self.cache.read_partition("short_interest", "all")
        if df.empty:
            return []
        df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
        clock_d = self.clock.now.date()
        mask = (df["ticker"] == ticker) & (df["settlement_date"] <= clock_d)
        if since is not None:
            mask = mask & (df["settlement_date"] >= since)
        return [
            ShortInterest(
                ticker=row["ticker"],
                settlement_date=row["settlement_date"],
                si_shares=int(row["si_shares"]),
                si_pct_float=float(row.get("si_pct_float", 0.0)),
                avg_daily_volume_20d=int(row["avg_daily_volume_20d"]),
            )
            for _, row in df[mask].iterrows()
        ]

    async def fetch_earnings(
        self: BacktestProvider, ticker: str, since: date | None = None
    ) -> list[EarningsEvent]:
        df = self.cache.read_partition("earnings", "all")
        if df.empty:
            return []
        df["report_at"] = pd.to_datetime(df["report_at"], utc=True)
        mask = (df["ticker"] == ticker) & (df["report_at"] <= self.clock.now)
        if since is not None:
            mask = mask & (df["report_at"].dt.date >= since)
        return [
            EarningsEvent(
                ticker=row["ticker"],
                report_at=row["report_at"].to_pydatetime(),
                actual_eps=row.get("actual_eps"),
                estimate_eps=row.get("estimate_eps"),
            )
            for _, row in df[mask].iterrows()
        ]

    async def fetch_sentiment(
        self: BacktestProvider, ticker: str, as_of: datetime
    ) -> RedditMention | None:
        df = self.cache.read_partition("sentiment", as_of.date().isoformat())
        if df.empty:
            return None
        df = df[df["ticker"] == ticker]
        if df.empty:
            return None
        row = df.iloc[-1]
        return RedditMention(
            ticker=ticker,
            as_of=as_of,
            subreddit=row["subreddit"],
            count_24h=int(row["count_24h"]),
            baseline_30d_mean=float(row["baseline_30d_mean"]),
            baseline_30d_std=float(row["baseline_30d_std"]),
        )
