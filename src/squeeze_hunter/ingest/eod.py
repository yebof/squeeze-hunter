"""Daily EOD ingest (P6): keep the parquet cache current for paper / live.

Until now the nightly scan in paper / live read whatever the operator had
last backfilled by hand. This job runs at 17:00 ET (`scheduler.ingest_eod`):

- bars: for every ticker, fetch from the day after the last cached bar to
  now (Yahoo) and append. One ticker failing never blocks the others.
- short interest: re-run the FINRA backfill when the last ingest is older
  than `data.finra_refresh_days` (the CDN → API fallback lives in the
  backfill); a fully failed download is reported, not raised.
- earnings: re-run the Finnhub backfill when older than
  `data.earnings_refresh_days`; skipped (and reported) without a key.

Every dataset writes a freshness stamp the premarket gate reads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

import pandas as pd

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.schema import Bar
from squeeze_hunter.ingest.backfill_earnings import backfill_earnings
from squeeze_hunter.ingest.backfill_finra import backfill_finra
from squeeze_hunter.ingest.freshness import record_freshness, recorded_age_days
from squeeze_hunter.logging_setup import get_logger

log = get_logger("ingest.eod")

# Transient / data-shaped failures a daily job must survive per ticker.
# AttributeError / TypeError / NotImplementedError still propagate (bugs).
_INGEST_ERRORS = (ConnectionError, TimeoutError, OSError, ValueError, KeyError, RuntimeError)


class BarSource(Protocol):
    """The one provider method the EOD job needs (YahooProvider satisfies it)."""

    async def fetch_bars(
        self, ticker: str, start: datetime, end: datetime, resolution: str = "1d"
    ) -> list[Bar]: ...


@dataclass
class EodIngestReport:
    bars_updated: dict[str, int] = field(default_factory=dict)
    bars_failed: list[str] = field(default_factory=list)
    finra: str = "skipped"
    earnings: str = "skipped"

    @property
    def ok(self) -> bool:
        return (
            not self.bars_failed
            and not self.finra.startswith("failed")
            and not self.earnings.startswith("failed")
        )


def _latest_bar_ts(cache: ParquetCache, ticker: str) -> datetime | None:
    df = cache.read_partition("bars", ticker)
    if df.empty:
        return None
    ts = pd.to_datetime(df["ts"], utc=True).max()
    return ts.to_pydatetime() if pd.notna(ts) else None


async def _ingest_bars(
    tickers: list[str],
    cache: ParquetCache,
    settings: Settings,
    now: datetime,
    yahoo: BarSource,
    report: EodIngestReport,
) -> None:
    newest_overall: datetime | None = None
    for t in tickers:
        last = _latest_bar_ts(cache, t)
        start = (
            (last + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            if last is not None
            else now - timedelta(days=settings.data.bars_backfill_days)
        )
        if start.date() > now.date():
            if last is not None:
                newest_overall = max(newest_overall or last, last)
            continue
        try:
            bars = await yahoo.fetch_bars(t, start, now)
        except _INGEST_ERRORS as e:
            log.warning("eod_bars_failed", ticker=t, err=str(e), err_type=type(e).__name__)
            report.bars_failed.append(t)
            continue
        if bars:
            df = pd.DataFrame(
                [
                    {
                        "ticker": b.ticker,
                        "ts": b.ts,
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                    }
                    for b in bars
                ]
            )
            cache.append_partition("bars", t, df, dedup_keys=["ticker", "ts"])
            report.bars_updated[t] = len(bars)
            newest_ticker = max(b.ts for b in bars)
        else:
            log.info("eod_bars_none", ticker=t, start=start.isoformat())
            newest_ticker = last
        if newest_ticker is not None:
            newest_overall = max(newest_overall or newest_ticker, newest_ticker)
    if newest_overall is not None:
        record_freshness(
            cache.root,
            "bars",
            as_of=newest_overall,
            rows=sum(report.bars_updated.values()),
            now=now,
            note=f"failed={report.bars_failed}" if report.bars_failed else None,
        )


def _latest_settlement(cache: ParquetCache) -> datetime | None:
    df = cache.read_partition("short_interest", "all")
    if df.empty:
        return None
    d = pd.to_datetime(df["settlement_date"]).max()
    return d.to_pydatetime().replace(tzinfo=UTC) if pd.notna(d) else None


def _latest_report(cache: ParquetCache) -> datetime | None:
    df = cache.read_partition("earnings", "all")
    if df.empty:
        return None
    d = pd.to_datetime(df["report_at"], utc=True).max()
    return d.to_pydatetime() if pd.notna(d) else None


async def ingest_eod(
    tickers: list[str],
    cache: ParquetCache,
    settings: Settings,
    now: datetime,
    *,
    yahoo: BarSource | None = None,
) -> EodIngestReport:
    report = EodIngestReport()
    source: BarSource
    if yahoo is None:
        from squeeze_hunter.data.providers.yahoo import YahooProvider

        source = YahooProvider()
    else:
        source = yahoo
    await _ingest_bars(tickers, cache, settings, now, source, report)

    # --- FINRA short interest (biweekly; cheap trailing-window bulk fetch)
    age = recorded_age_days(cache.root, "short_interest", now)
    if age is not None and age < settings.data.finra_refresh_days:
        report.finra = "skipped:fresh"
    else:
        try:
            await backfill_finra(tickers, cache)
            latest = _latest_settlement(cache)
            record_freshness(
                cache.root,
                "short_interest",
                as_of=latest or now,
                rows=len(cache.read_partition("short_interest", "all")),
                now=now,
            )
            report.finra = "updated"
        except _INGEST_ERRORS as e:
            log.error("eod_finra_failed", err=str(e), err_type=type(e).__name__)
            report.finra = f"failed:{type(e).__name__}: {e}"

    # --- Earnings calendar (weekly; needs FINNHUB_KEY)
    if not os.environ.get("FINNHUB_KEY"):
        report.earnings = "skipped:no_key"
    else:
        age = recorded_age_days(cache.root, "earnings", now)
        if age is not None and age < settings.data.earnings_refresh_days:
            report.earnings = "skipped:fresh"
        else:
            try:
                await backfill_earnings(tickers, cache)
                record_freshness(
                    cache.root,
                    "earnings",
                    as_of=_latest_report(cache) or now,
                    rows=len(cache.read_partition("earnings", "all")),
                    now=now,
                )
                report.earnings = "updated"
            except _INGEST_ERRORS as e:
                log.error("eod_earnings_failed", err=str(e), err_type=type(e).__name__)
                report.earnings = f"failed:{type(e).__name__}: {e}"

    log.info(
        "eod_ingest_complete",
        bars_updated=sum(report.bars_updated.values()),
        bars_failed=report.bars_failed,
        finra=report.finra,
        earnings=report.earnings,
    )
    return report
