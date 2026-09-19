"""P6 — FINRA Query API client, and the CDN → API fallback in the backfill."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.data.providers.finra_api import FinraApiClient
from squeeze_hunter.ingest.backfill_finra import backfill_finra


def _transport(pages: list[list[dict]]) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    page_iter = iter(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/oauth2/access_token"):
            assert request.headers["Authorization"].startswith("Basic ")
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 1800})
        assert request.headers["Authorization"] == "Bearer tok"
        body = json.loads(request.content)
        assert body["compareFilters"][0]["fieldValue"] == "GME"
        return httpx.Response(200, json=next(page_iter, []))

    return httpx.MockTransport(handler), seen


@pytest.mark.asyncio
async def test_client_authenticates_pages_and_parses() -> None:
    row = {
        "settlementDate": "2024-04-30",
        "symbolCode": "GME",
        "currentShortPositionQuantity": 10000000,
        "averageDailyVolumeQuantity": 2000000,
    }
    row2 = dict(row, settlementDate="2024-04-15", currentShortPositionQuantity=9000000)
    transport, seen = _transport([[row, row2], []])
    client = FinraApiClient(client_id="id", client_secret="s", transport=transport, page_size=2)
    out = await client.fetch_short_interest(["GME"], since=date(2024, 1, 1))
    got = sorted(out["GME"], key=lambda r: r.settlement_date)
    assert [(r.settlement_date, r.si_shares, r.avg_daily_volume_20d) for r in got] == [
        (date(2024, 4, 15), 9000000, 2000000),
        (date(2024, 4, 30), 10000000, 2000000),
    ]
    assert all(r.si_pct_float == 0.0 for r in got)
    # one token call + two data pages
    assert len(seen) == 3
    assert json.loads(seen[2].content)["offset"] == 2


@pytest.mark.asyncio
async def test_backfill_falls_back_to_the_api_when_the_cdn_is_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FINRA_API_CLIENT_ID", "id")
    monkeypatch.setenv("FINRA_API_CLIENT_SECRET", "s")
    from squeeze_hunter.data.schema import ShortInterest

    api_rows = {
        "GME": [
            ShortInterest(
                ticker="GME",
                settlement_date=date(2024, 4, 30),
                si_shares=10_000_000,
                si_pct_float=0.0,
                avg_daily_volume_20d=2_000_000,
            )
        ]
    }
    with (
        patch(
            "squeeze_hunter.ingest.backfill_finra.FinraProvider.fetch_short_interest_bulk",
            new=AsyncMock(side_effect=RuntimeError("0 of 210 FINRA monthly files downloaded")),
        ),
        patch(
            "squeeze_hunter.ingest.backfill_finra.FinraApiClient.fetch_short_interest",
            new=AsyncMock(return_value=api_rows),
        ) as api,
        patch(
            "squeeze_hunter.ingest.backfill_finra.YahooProvider.get_float_shares",
            new=AsyncMock(return_value=50_000_000),
        ),
        patch(
            "squeeze_hunter.ingest.backfill_finra.YahooProvider.get_split_ratios",
            new=AsyncMock(return_value=[]),
        ),
    ):
        await backfill_finra(["GME"], ParquetCache(root=tmp_path))
    api.assert_awaited_once()
    out = ParquetCache(root=tmp_path).read_partition("short_interest", "all")
    assert out["si_pct_float"].iloc[0] == pytest.approx(0.20)


@pytest.mark.asyncio
async def test_backfill_without_api_credentials_still_fails_loud(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("FINRA_API_CLIENT_ID", raising=False)
    monkeypatch.delenv("FINRA_API_CLIENT_SECRET", raising=False)
    with (
        patch(
            "squeeze_hunter.ingest.backfill_finra.FinraProvider.fetch_short_interest_bulk",
            new=AsyncMock(side_effect=RuntimeError("0 of 210 FINRA monthly files downloaded")),
        ),
        pytest.raises(RuntimeError, match="FINRA"),
    ):
        await backfill_finra(["GME"], ParquetCache(root=tmp_path))
