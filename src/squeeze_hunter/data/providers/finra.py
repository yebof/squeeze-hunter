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

    @staticmethod
    def _candidate_months(since: date | None) -> list[tuple[str, str]]:
        """CDX-P2-5: the (YYYYMM, half) report files spanning `since`→today.

        `since=None` keeps the lightweight trailing-3-month window for live
        use; a historical backfill passes an explicit `since` and gets every
        month back to it.
        """
        today = date.today()
        if since is None:
            n_months = 3
        else:
            n_months = (today.year - since.year) * 12 + (today.month - since.month) + 1
            n_months = max(1, n_months)
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
        return candidates

    async def fetch_short_interest(
        self: FinraProvider,
        ticker: str,
        since: date | None = None,
    ) -> list[ShortInterest]:
        """Single-ticker fetch (DataProvider Protocol). Downloads every report
        from `since` to today and filters to `ticker`.

        NOTE: for a multi-ticker backfill use `fetch_short_interest_bulk` --
        calling this once per ticker re-downloads the SAME ~N monthly files
        for every ticker (CDX2-P2: tickers x months duplicate GETs, slow /
        rate-limited). This single-ticker form is fine for occasional
        one-off lookups.
        """
        results: list[ShortInterest] = []
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for yyyymm, half in self._candidate_months(since):
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

    async def fetch_short_interest_bulk(
        self: FinraProvider,
        tickers: list[str],
        since: date | None = None,
    ) -> dict[str, list[ShortInterest]]:
        """CDX2-P2: download each monthly report file EXACTLY ONCE and index
        the requested tickers from it.

        The prior backfill called the single-ticker `fetch_short_interest`
        per ticker, so a ~1500-name universe over ~7 years re-downloaded the
        same ~180 FINRA files ~1500 times (~270k GETs of identical files) --
        slow and a near-certain rate-limit/ban. This streams each file once
        and filters by the requested universe, bounding both HTTP volume
        (~one GET per file) and memory (only `tickers` are retained).
        """
        wanted = set(tickers)
        out: dict[str, list[ShortInterest]] = {t: [] for t in tickers}
        candidates = self._candidate_months(since)
        n_ok = 0
        last_error = ""
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            for yyyymm, half in candidates:
                url = FINRA_URL_TEMPLATE.format(yyyymm=yyyymm, half=half)
                try:
                    r = await client.get(url)
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    # Individual months can legitimately be missing (the
                    # current half-month, a not-yet-published file); keep going.
                    last_error = f"{url}: {type(e).__name__} {e}"
                    continue
                n_ok += 1
                for row in self._parse_finra_pipe_text(StringIO(r.text)):
                    if row.ticker in wanted and (since is None or row.settlement_date >= since):
                        out[row.ticker].append(row)
        if candidates and n_ok == 0:
            # Round-12: every file failed (observed: cdn.finra.org → 403 for all
            # 210 months). Swallowing that produced an exit-0 backfill that
            # wrote nothing, leaving f1/f2 silently dead. Fail loud instead.
            raise RuntimeError(
                f"0 of {len(candidates)} FINRA monthly files downloaded; last error: "
                f"{last_error}. Check network access to cdn.finra.org (403 = blocked "
                "or the URL scheme changed) before trusting any backtest."
            )
        return out

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
