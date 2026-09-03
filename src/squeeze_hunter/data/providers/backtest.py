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
from squeeze_hunter.logging_setup import get_logger

log = get_logger("data.backtest")


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
    # R11: FINRA disseminates a settlement-date short-interest report ~8 US
    # business days later. fetch_short_interest reveals each record on
    # settlement_date + this many business days so the backtest can't act on SI
    # before it was public (lookahead). Wired from settings.data in production.
    finra_publication_lag_bdays: int = 8
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
        if end > self.clock.now:
            log.warning(
                "backtest_end_exceeds_clock",
                ticker=ticker,
                end=end.isoformat(),
                clock=self.clock.now.isoformat(),
            )
        df = self.cache.read_partition("bars", ticker)
        if df.empty:
            log.info("backtest_no_bars_partition", ticker=ticker)
            return []
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        mask = (df["ts"] >= start) & (df["ts"] <= end) & (df["ts"] <= self.clock.now)
        bars = []
        for _, row in df[mask].iterrows():
            o = float(row["open"])
            h = float(row["high"])
            lo = float(row["low"])
            c = float(row["close"])
            # Clamp high/low so OHLC constraints hold even for noisy cached data.
            h = max(h, o, c)
            lo = min(lo, o, c)
            bars.append(
                Bar(
                    ticker=row["ticker"],
                    ts=row["ts"].to_pydatetime(),
                    open=o,
                    high=h,
                    low=lo,
                    close=c,
                    volume=int(row["volume"]),
                )
            )
        return bars

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
        # R9.11: partition key format is `{ticker}__{YYYY-MM-DD}`. There is NO
        # writer for this partition in src/ today — Phase 4 must add an
        # options-chain ingest job that writes via cache.append_partition with
        # this exact key format. Otherwise f5 (call OI velocity) returns 0
        # for every ticker and Gate 1 cannot validate the GME setup.
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

    async def fetch_option_chain_at(
        self: BacktestProvider, ticker: str, as_of: datetime
    ) -> OptionChain:
        """Read a historical option chain from cache.

        Reads the parquet partition for the requested date. If absent, returns an
        empty OptionChain (so the caller distinguishes "no data" from "real zeros").
        Respects the clock — cannot read future dates.

        R9.11: shares the partition key format `{ticker}__{YYYY-MM-DD}` with
        fetch_option_chain — keep them in lockstep when writing the Phase 4
        ingest job.
        """
        if as_of > self.clock.now:
            return OptionChain(underlying=ticker, as_of=as_of, spot=0.0, quotes=[])
        df = self.cache.read_partition("options", f"{ticker}__{as_of.date().isoformat()}")
        if df.empty:
            return OptionChain(underlying=ticker, as_of=as_of, spot=0.0, quotes=[])
        from squeeze_hunter.data.schema import OptionQuote

        quotes = [
            OptionQuote(
                strike=float(r["strike"]),
                expiry=date.fromisoformat(str(r["expiry"])),
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
        return OptionChain(underlying=ticker, as_of=as_of, spot=spot, quotes=quotes)

    async def fetch_short_interest(
        self: BacktestProvider, ticker: str, since: date | None = None
    ) -> list[ShortInterest]:
        df = self.cache.read_partition("short_interest", "all")
        if df.empty:
            return []
        df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
        clock_d = self.clock.now.date()
        # R11: gate visibility on the AVAILABILITY date (settlement + FINRA
        # publication lag), not the settlement date. FINRA publishes a
        # settlement-date report ~8 business days later, so revealing it on the
        # settlement date lets the backtest act on short interest the live
        # system could not yet have known — lookahead that inflates Gate 1.
        lag = self.finra_publication_lag_bdays
        if lag > 0:
            # Round-12: count US federal holidays like the rest of the codebase
            # (time stop, f3 window). Plain BusinessDay revealed the record one
            # business day early whenever a holiday fell inside the lag window.
            from squeeze_hunter.signals.earnings_reaction import _us_business_holidays

            bday = pd.offsets.CustomBusinessDay(n=lag, holidays=_us_business_holidays())
            available = (pd.to_datetime(df["settlement_date"]) + bday).dt.date
        else:
            available = df["settlement_date"]
        mask = (df["ticker"] == ticker) & (available <= clock_d)
        # `since` filters by settlement date (it selects how far back to look,
        # which is a property of the report period, not its availability).
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
