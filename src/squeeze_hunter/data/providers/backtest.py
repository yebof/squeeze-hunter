"""BacktestProvider — replays cached parquet history with a clock that
prevents lookahead bias by construction."""

from __future__ import annotations

import bisect
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

    # P7: read each parquet partition ONCE per provider and keep it prepared
    # (parsed timestamps, chronological order, FINRA availability dates). The
    # runner used to re-read and re-parse the same files for every ticker on
    # every session — the dominant cost of a multi-year backtest. A provider
    # lives for one backtest run / one nightly scan, so staleness is a
    # non-issue; call `invalidate()` after writing to the cache mid-run.
    _frames: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict, repr=False)

    def invalidate(self: BacktestProvider) -> None:
        self._frames.clear()
        self._bars.clear()

    _bars: dict[str, tuple[list[Bar], list[datetime]]] = field(default_factory=dict, repr=False)

    def _bars_prepared(self: BacktestProvider, ticker: str) -> tuple[list[Bar], list[datetime]]:
        """All bars for `ticker`, chronological, as Bar objects built once
        (pydantic construction per call was the next cost after parquet I/O)."""
        if ticker not in self._bars:
            df = self.cache.read_partition("bars", ticker)
            bars: list[Bar] = []
            if not df.empty:
                df = df.copy()
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                # Round-13: chronological regardless of parquet storage order.
                df = df.sort_values("ts", kind="stable").reset_index(drop=True)
                columns = (
                    df["ticker"],
                    df["ts"],
                    df["open"],
                    df["high"],
                    df["low"],
                    df["close"],
                    df["volume"],
                )
                for ticker_v, ts_v, o_v, h_v, lo_v, c_v, vol_v in zip(*columns, strict=True):
                    ts_py = pd.Timestamp(ts_v).to_pydatetime()
                    if not isinstance(ts_py, datetime) or pd.isna(ts_v):
                        continue  # NaT: unusable row
                    o = float(o_v)
                    h = float(h_v)
                    lo = float(lo_v)
                    c = float(c_v)
                    # Clamp high/low so OHLC constraints hold even for noisy cached data.
                    bars.append(
                        Bar(
                            ticker=str(ticker_v),
                            ts=ts_py,
                            open=o,
                            high=max(h, o, c),
                            low=min(lo, o, c),
                            close=c,
                            volume=int(vol_v),
                        )
                    )
            self._bars[ticker] = (bars, [b.ts for b in bars])
        return self._bars[ticker]

    def _short_interest_frame(self: BacktestProvider) -> pd.DataFrame:
        key = ("short_interest", "all")
        if key not in self._frames:
            df = self.cache.read_partition("short_interest", "all")
            if not df.empty:
                df = df.copy()
                df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
                lag = self.finra_publication_lag_bdays
                if lag > 0:
                    # R11 + Round-13: reveal on settlement + lag NYSE sessions.
                    from squeeze_hunter.trading_calendar import nyse_holidays

                    bday = pd.offsets.CustomBusinessDay(n=lag, holidays=nyse_holidays())
                    distinct = pd.to_datetime(pd.Series(df["settlement_date"].unique()))
                    avail = dict(zip(distinct.dt.date, (distinct + bday).dt.date, strict=True))
                    df["available"] = df["settlement_date"].map(avail)
                else:
                    df["available"] = df["settlement_date"]
            self._frames[key] = df
        return self._frames[key]

    def _earnings_frame(self: BacktestProvider) -> pd.DataFrame:
        key = ("earnings", "all")
        if key not in self._frames:
            df = self.cache.read_partition("earnings", "all")
            if not df.empty:
                df = df.copy()
                df["report_at"] = pd.to_datetime(df["report_at"], utc=True)
            self._frames[key] = df
        return self._frames[key]

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
        bars, stamps = self._bars_prepared(ticker)
        if not bars:
            log.info("backtest_no_bars_partition", ticker=ticker)
            return []
        hi = min(end, self.clock.now)
        lo_idx = bisect.bisect_left(stamps, start)
        hi_idx = bisect.bisect_right(stamps, hi)
        return bars[lo_idx:hi_idx]

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
        df = self._short_interest_frame()
        if df.empty:
            return []
        clock_d = self.clock.now.date()
        available = df["available"]
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
        df = self._earnings_frame()
        if df.empty:
            return []
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
