from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.schema import ShortInterest
from squeeze_hunter.ingest.backfill_finra import backfill_finra


@pytest.mark.asyncio
async def test_backfill_finra_merges_float_into_si_pct(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    fake_si = [
        ShortInterest(
            ticker="GME",
            settlement_date=date(2024, 4, 30),
            si_shares=10_000_000,
            si_pct_float=0.0,
            avg_daily_volume_20d=2_000_000,
        )
    ]
    with (
        patch(
            "squeeze_hunter.ingest.backfill_finra.FinraProvider.fetch_short_interest",
            new=AsyncMock(return_value=fake_si),
        ),
        patch(
            "squeeze_hunter.ingest.backfill_finra.YahooProvider.get_float_shares",
            new=AsyncMock(return_value=50_000_000),  # 50M float
        ),
    ):
        await backfill_finra(["GME"], cache)
    out = cache.read_partition("short_interest", "all")
    assert len(out) == 1
    # 10M / 50M = 0.20
    assert out["si_pct_float"].iloc[0] == pytest.approx(0.20)


@pytest.mark.asyncio
async def test_backfill_finra_no_float_keeps_zero(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    fake_si = [
        ShortInterest(
            ticker="GME",
            settlement_date=date(2024, 4, 30),
            si_shares=10_000_000,
            si_pct_float=0.0,
            avg_daily_volume_20d=2_000_000,
        )
    ]
    with (
        patch(
            "squeeze_hunter.ingest.backfill_finra.FinraProvider.fetch_short_interest",
            new=AsyncMock(return_value=fake_si),
        ),
        patch(
            "squeeze_hunter.ingest.backfill_finra.YahooProvider.get_float_shares",
            new=AsyncMock(return_value=None),
        ),
    ):
        await backfill_finra(["GME"], cache)
    out = cache.read_partition("short_interest", "all")
    assert out["si_pct_float"].iloc[0] == 0.0


@pytest.mark.asyncio
async def test_backfill_finra_yahoo_exception_does_not_abort(tmp_path: Path) -> None:
    """If Yahoo raises for one ticker, others should still ingest."""
    cache = ParquetCache(root=tmp_path)
    si_by_ticker = {
        "BAD": [
            ShortInterest(
                ticker="BAD",
                settlement_date=date(2024, 4, 30),
                si_shares=1_000_000,
                si_pct_float=0.0,
                avg_daily_volume_20d=100_000,
            )
        ],
        "GOOD": [
            ShortInterest(
                ticker="GOOD",
                settlement_date=date(2024, 4, 30),
                si_shares=2_000_000,
                si_pct_float=0.0,
                avg_daily_volume_20d=200_000,
            )
        ],
    }

    async def fake_si_fetch(self, ticker, since=None):
        return si_by_ticker.get(ticker, [])

    async def fake_float_fetch(self, ticker):
        if ticker == "BAD":
            # R8.M11 narrowed the catch to actual yfinance failure modes.
            # ConnectionError is what httpx raises on transient network failures
            # and is in the narrowed catch list.
            raise ConnectionError("yahoo down")
        return 10_000_000  # GOOD: 2M / 10M = 0.20

    with (
        patch(
            "squeeze_hunter.ingest.backfill_finra.FinraProvider.fetch_short_interest",
            new=fake_si_fetch,
        ),
        patch(
            "squeeze_hunter.ingest.backfill_finra.YahooProvider.get_float_shares",
            new=fake_float_fetch,
        ),
    ):
        await backfill_finra(["BAD", "GOOD"], cache)
    out = cache.read_partition("short_interest", "all")
    by_ticker = out.set_index("ticker")
    assert by_ticker.loc["BAD", "si_pct_float"] == 0.0  # fallback
    assert by_ticker.loc["GOOD", "si_pct_float"] == pytest.approx(0.20)
