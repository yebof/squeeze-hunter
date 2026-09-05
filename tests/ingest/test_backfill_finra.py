from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.finra import FinraProvider
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
            "squeeze_hunter.ingest.backfill_finra.FinraProvider.fetch_short_interest_bulk",
            new=AsyncMock(return_value={"GME": fake_si}),
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
            "squeeze_hunter.ingest.backfill_finra.FinraProvider.fetch_short_interest_bulk",
            new=AsyncMock(return_value={"GME": fake_si}),
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

    async def fake_bulk(self, tickers, since=None):
        return {t: si_by_ticker.get(t, []) for t in tickers}

    async def fake_float_fetch(self, ticker):
        if ticker == "BAD":
            # R8.M11 narrowed the catch to actual yfinance failure modes.
            # ConnectionError is what httpx raises on transient network failures
            # and is in the narrowed catch list.
            raise ConnectionError("yahoo down")
        return 10_000_000  # GOOD: 2M / 10M = 0.20

    with (
        patch(
            "squeeze_hunter.ingest.backfill_finra.FinraProvider.fetch_short_interest_bulk",
            new=fake_bulk,
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


@pytest.mark.asyncio
async def test_backfill_finra_downloads_each_monthly_file_once(tmp_path: Path) -> None:
    """CDX2-P2 regression: the prior backfill called fetch_short_interest once
    PER TICKER, and each call re-downloaded EVERY monthly file. With a real
    universe that is tickers x months identical GETs (slow, rate-limited).
    fetch_short_interest_bulk must GET each unique monthly file exactly once
    regardless of how many tickers are requested.
    """
    cache = ParquetCache(root=tmp_path)
    requested_urls: list[str] = []

    sample = (
        "Settlement Date|Symbol Code|Symbol|Market Class Code|Current Short Position|"
        "Previous Short Position|Change|Percent Change|Average Daily Volume|"
        "Days To Cover|Revision Indicator\n"
        "20240430|GME|GME|N|10000000|9500000|500000|5.26|2000000|5.0|N\n"
        "20240430|AMC|AMC|N|20000000|19000000|1000000|5.26|3000000|6.0|N\n"
    )

    class _Resp:
        text = sample

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def __init__(self, *a, **kw) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *a) -> None:
            return None

        async def get(self, url: str) -> _Resp:
            requested_urls.append(url)
            return _Resp()

    tickers = ["GME", "AMC", "BBBY", "KOSS", "BYND"]
    expected_unique = len(FinraProvider._candidate_months(date(2018, 1, 1)))

    with (
        patch.object(httpx, "AsyncClient", _Client),
        patch(
            "squeeze_hunter.ingest.backfill_finra.YahooProvider.get_float_shares",
            new=AsyncMock(return_value=10_000_000),
        ),
    ):
        await backfill_finra(tickers, cache)

    # Exactly one GET per unique monthly file — NOT multiplied by ticker count.
    assert len(requested_urls) == expected_unique, (
        f"{len(requested_urls)} GETs for {expected_unique} files across "
        f"{len(tickers)} tickers — bulk path should download each file once"
    )
    assert len(requested_urls) == len(set(requested_urls)), "duplicate file downloads"

    # And the data still lands correctly for the requested universe.
    out = cache.read_partition("short_interest", "all")
    assert set(out["ticker"]) <= set(tickers)
    assert (out["ticker"] == "GME").any()


@pytest.fixture(autouse=True)
def _no_splits():
    """Round-13: backfill_finra now also asks Yahoo for split history; keep these
    tests hermetic (no network) by defaulting it to 'no splits'."""
    with patch(
        "squeeze_hunter.ingest.backfill_finra.YahooProvider.get_split_ratios",
        new=AsyncMock(return_value=[]),
    ):
        yield
