"""P6 — the daily EOD ingest keeps the parquet cache current for paper/live."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.schema import Bar
from squeeze_hunter.ingest.eod import ingest_eod
from squeeze_hunter.ingest.freshness import read_freshness, record_freshness

_NOW = datetime(2026, 6, 10, 21, 30, tzinfo=UTC)  # 17:30 ET


def _bar(ticker: str, ts: datetime, close: float = 10.0) -> Bar:
    return Bar(ticker=ticker, ts=ts, open=close, high=close, low=close, close=close, volume=1000)


def _seed_bars(cache: ParquetCache, ticker: str, last_day: date, n: int = 5) -> None:
    rows = [
        {
            "ticker": ticker,
            "ts": datetime.combine(last_day - timedelta(days=i), datetime.min.time(), tzinfo=UTC)
            + timedelta(hours=4),
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1000,
        }
        for i in range(n)
    ]
    cache.write_partition("bars", ticker, pd.DataFrame(rows))


class _FakeYahoo:
    def __init__(self, bars_by_ticker: dict[str, list[Bar]], fail: set[str] = frozenset()) -> None:
        self.bars_by_ticker = bars_by_ticker
        self.fail = fail
        self.calls: list[tuple[str, datetime, datetime]] = []

    async def fetch_bars(self, ticker: str, start: datetime, end: datetime, resolution: str = "1d"):
        self.calls.append((ticker, start, end))
        if ticker in self.fail:
            raise ConnectionError("yahoo down")
        return [b for b in self.bars_by_ticker.get(ticker, []) if start <= b.ts <= end]


@pytest.mark.asyncio
async def test_bars_are_fetched_incrementally_from_the_last_cached_session(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed_bars(cache, "GME", date(2026, 6, 8))  # Monday; today is Wednesday
    new = [
        _bar("GME", datetime(2026, 6, 9, 4, tzinfo=UTC), 11.0),
        _bar("GME", datetime(2026, 6, 10, 4, tzinfo=UTC), 12.0),
    ]
    yahoo = _FakeYahoo({"GME": new})
    report = await ingest_eod(["GME"], cache, Settings(), _NOW, yahoo=yahoo)  # type: ignore[arg-type]
    ticker, start, _end = yahoo.calls[0]
    assert ticker == "GME"
    assert start.date() == date(2026, 6, 9)  # the day AFTER the last cached bar
    out = cache.read_partition("bars", "GME")
    assert len(out) == 7
    assert report.bars_updated == {"GME": 2}
    stamps = read_freshness(tmp_path)
    assert stamps["bars"]["as_of"].startswith("2026-06-10")


@pytest.mark.asyncio
async def test_one_failing_ticker_does_not_block_the_others(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed_bars(cache, "GME", date(2026, 6, 9))
    _seed_bars(cache, "AMC", date(2026, 6, 9))
    yahoo = _FakeYahoo({"AMC": [_bar("AMC", datetime(2026, 6, 10, 4, tzinfo=UTC))]}, fail={"GME"})
    report = await ingest_eod(["GME", "AMC"], cache, Settings(), _NOW, yahoo=yahoo)  # type: ignore[arg-type]
    assert report.bars_failed == ["GME"]
    assert report.bars_updated == {"AMC": 1}


@pytest.mark.asyncio
async def test_finra_and_earnings_refresh_only_when_due(tmp_path: Path, monkeypatch) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed_bars(cache, "GME", date(2026, 6, 10))
    record_freshness(
        tmp_path,
        "short_interest",
        as_of=_NOW - timedelta(days=3),
        rows=1,
        now=_NOW - timedelta(days=1),
    )
    record_freshness(
        tmp_path, "earnings", as_of=_NOW - timedelta(days=3), rows=1, now=_NOW - timedelta(days=1)
    )
    monkeypatch.setenv("FINNHUB_KEY", "k")
    finra = AsyncMock()
    earnings = AsyncMock()
    with (
        patch("squeeze_hunter.ingest.eod.backfill_finra", finra),
        patch("squeeze_hunter.ingest.eod.backfill_earnings", earnings),
    ):
        report = await ingest_eod(["GME"], cache, Settings(), _NOW, yahoo=_FakeYahoo({}))  # type: ignore[arg-type]
    finra.assert_not_awaited()
    earnings.assert_not_awaited()
    assert report.finra == "skipped:fresh"
    assert report.earnings == "skipped:fresh"

    # Eight days later both are due; FINRA fails loud, earnings succeeds.
    later = _NOW + timedelta(days=8)
    finra = AsyncMock(side_effect=RuntimeError("0 of 6 FINRA monthly files downloaded"))
    earnings = AsyncMock()
    with (
        patch("squeeze_hunter.ingest.eod.backfill_finra", finra),
        patch("squeeze_hunter.ingest.eod.backfill_earnings", earnings),
    ):
        report = await ingest_eod(["GME"], cache, Settings(), later, yahoo=_FakeYahoo({}))  # type: ignore[arg-type]
    finra.assert_awaited_once()
    earnings.assert_awaited_once()
    assert report.finra.startswith("failed:")
    assert report.earnings == "updated"


@pytest.mark.asyncio
async def test_earnings_refresh_is_skipped_without_a_key(tmp_path: Path, monkeypatch) -> None:
    cache = ParquetCache(root=tmp_path)
    monkeypatch.delenv("FINNHUB_KEY", raising=False)
    with patch("squeeze_hunter.ingest.eod.backfill_finra", AsyncMock()):
        report = await ingest_eod(["GME"], cache, Settings(), _NOW, yahoo=_FakeYahoo({}))  # type: ignore[arg-type]
    assert report.earnings == "skipped:no_key"
