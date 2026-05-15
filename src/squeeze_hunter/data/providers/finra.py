"""FinraProvider — biweekly short-interest bulk file (CDN, not FTP since 2023)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import StringIO
from typing import IO

import httpx

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    Quote,
    RedditMention,
    ShortInterest,
)

# FINRA publishes biweekly short interest at:
# https://cdn.finra.org/equity/regsho/monthly/shrt{YYYYMM}{a|b}.txt
# where the suffix is 'a' for the 15th-of-month report and 'b' for the
# end-of-month report. Format is pipe-delimited with the columns parsed below.
FINRA_URL_TEMPLATE = "https://cdn.finra.org/equity/regsho/monthly/shrt{yyyymm}{half}.txt"


@dataclass
class FinraProvider:
    name: str = "finra"
    capabilities: frozenset[str] = frozenset({"si"})
    timeout_s: float = 30.0

    async def fetch_short_interest(
        self: FinraProvider,
        ticker: str,
        since: date | None = None,
    ) -> list[ShortInterest]:
        """Fetch every biweekly report from `since` to today.

        CDX-P2-5: this previously hardcoded the trailing ~3 months
        (`range(0, 3)`), so a historical backfill that passes
        `since=date(2018, 1, 1)` only ever got the last quarter of data and
        f1/f2 were dead for years of any multi-year backtest. We now iterate
        EVERY month from `since` to the current month. When `since` is None
        (live/no-history use) we keep the lightweight trailing-3-month window.
        """
        today = date.today()
        # Inclusive month count from `since` to today. None → trailing 3.
        if since is None:
            n_months = 3
        else:
            n_months = (today.year - since.year) * 12 + (today.month - since.month) + 1
            n_months = max(1, n_months)
        results: list[ShortInterest] = []
        candidates: list[tuple[str, str]] = []
        for months_back in range(0, n_months):
            year = today.year
            month = today.month - months_back
            while month <= 0:
                month += 12
                year -= 1
            yyyymm = f"{year:04d}{month:02d}"
            for half in ("b", "a"):
                candidates.append((yyyymm, half))
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for yyyymm, half in candidates:
                url = FINRA_URL_TEMPLATE.format(yyyymm=yyyymm, half=half)
                try:
                    r = await client.get(url)
                    r.raise_for_status()
                except httpx.HTTPError:
                    continue
                for row in self._parse_finra_pipe_text(StringIO(r.text)):
                    if row.ticker == ticker and (since is None or row.settlement_date >= since):
                        results.append(row)
        return results

    @staticmethod
    def _parse_finra_pipe_text(stream: IO[str]) -> Iterator[ShortInterest]:
        header = stream.readline().rstrip("\n").split("|")
        idx = {h.strip(): i for i, h in enumerate(header)}
        for line in stream:
            parts = line.rstrip("\n").split("|")
            if len(parts) < len(header):
                continue
            try:
                yield ShortInterest(
                    ticker=parts[idx["Symbol"]],
                    settlement_date=datetime.strptime(
                        parts[idx["Settlement Date"]], "%Y%m%d"
                    ).date(),
                    si_shares=int(parts[idx["Current Short Position"]]),
                    si_pct_float=0.0,  # FINRA file has no float; merged later by signal layer
                    avg_daily_volume_20d=int(float(parts[idx["Average Daily Volume"]])),
                )
            except (ValueError, KeyError):
                continue

    async def fetch_bars(self: FinraProvider, *_a: object, **_kw: object) -> list[Bar]:
        return []

    async def fetch_quote(self: FinraProvider, ticker: str) -> Quote:
        raise NotImplementedError

    async def fetch_option_chain(self: FinraProvider, *_a: object, **_kw: object) -> OptionChain:
        return OptionChain(underlying="", as_of=datetime.now(UTC), spot=0.0, quotes=[])

    async def fetch_earnings(
        self: FinraProvider, *_a: object, **_kw: object
    ) -> list[EarningsEvent]:
        return []

    async def fetch_sentiment(
        self: FinraProvider, *_a: object, **_kw: object
    ) -> RedditMention | None:
        return None

    async def fetch_option_chain_at(
        self: FinraProvider, ticker: str, as_of: datetime
    ) -> OptionChain:
        """Live providers don't yet cache historical option chains. Returns empty."""
        return OptionChain(underlying=ticker, as_of=as_of, spot=0.0, quotes=[])
