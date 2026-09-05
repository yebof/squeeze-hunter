"""Round-13: FINRA short shares are as-reported (never restated for splits)
while Yahoo's float is today's. f1 = si_shares / float therefore needs the
historical share count expressed in TODAY's share basis."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.schema import ShortInterest
from squeeze_hunter.ingest.backfill_finra import backfill_finra


def _si(settlement: date, shares: int) -> ShortInterest:
    return ShortInterest(
        ticker="GME",
        settlement_date=settlement,
        si_shares=shares,
        si_pct_float=0.0,
        avg_daily_volume_20d=2_000_000,
    )


@pytest.mark.asyncio
async def test_si_pct_float_adjusts_pre_split_share_counts(tmp_path: Path) -> None:
    """GME 4:1 split on 2022-07-22 (today's float 200M): a 10M short position
    reported BEFORE the split is 40M of today's shares (20%); one reported
    after it is 10M (5%)."""
    cache = ParquetCache(root=tmp_path)
    records = [_si(date(2022, 6, 30), 10_000_000), _si(date(2022, 8, 15), 10_000_000)]
    with (
        patch(
            "squeeze_hunter.ingest.backfill_finra.FinraProvider.fetch_short_interest_bulk",
            new=AsyncMock(return_value={"GME": records}),
        ),
        patch(
            "squeeze_hunter.ingest.backfill_finra.YahooProvider.get_float_shares",
            new=AsyncMock(return_value=200_000_000),
        ),
        patch(
            "squeeze_hunter.ingest.backfill_finra.YahooProvider.get_split_ratios",
            new=AsyncMock(return_value=[(date(2022, 7, 22), 4.0)]),
        ),
    ):
        await backfill_finra(["GME"], cache)
    out = cache.read_partition("short_interest", "all").set_index("settlement_date")
    assert out.loc[date(2022, 6, 30), "si_pct_float"] == pytest.approx(0.20)
    assert out.loc[date(2022, 8, 15), "si_pct_float"] == pytest.approx(0.05)
    # The as-reported count is preserved; days_to_cover uses it unadjusted.
    assert int(out.loc[date(2022, 6, 30), "si_shares"]) == 10_000_000


@pytest.mark.asyncio
async def test_reverse_split_scales_pre_split_counts_down(tmp_path: Path) -> None:
    """AMC 1:10 reverse split (ratio 0.1): a pre-split 100M short is 10M today."""
    cache = ParquetCache(root=tmp_path)
    with (
        patch(
            "squeeze_hunter.ingest.backfill_finra.FinraProvider.fetch_short_interest_bulk",
            new=AsyncMock(return_value={"GME": [_si(date(2023, 6, 30), 100_000_000)]}),
        ),
        patch(
            "squeeze_hunter.ingest.backfill_finra.YahooProvider.get_float_shares",
            new=AsyncMock(return_value=100_000_000),
        ),
        patch(
            "squeeze_hunter.ingest.backfill_finra.YahooProvider.get_split_ratios",
            new=AsyncMock(return_value=[(date(2023, 8, 24), 0.1)]),
        ),
    ):
        await backfill_finra(["GME"], cache)
    out = cache.read_partition("short_interest", "all")
    assert out["si_pct_float"].iloc[0] == pytest.approx(0.10)
