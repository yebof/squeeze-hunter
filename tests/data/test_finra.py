from datetime import date
from io import StringIO

import httpx
import pytest

from squeeze_hunter.data.providers.finra import FinraProvider

SAMPLE_FILE = """Settlement Date|Symbol Code|Symbol|Market Class Code|Current Short Position|Previous Short Position|Change|Percent Change|Average Daily Volume|Days To Cover|Revision Indicator
20240430|GME|GME|N|10000000|9500000|500000|5.26|2000000|5.0|N
20240430|AAPL|AAPL|N|50000000|48000000|2000000|4.17|80000000|0.625|N
"""


@pytest.mark.asyncio
async def test_parse_finra_sample() -> None:
    p = FinraProvider()
    rows = list(p._parse_finra_pipe_text(StringIO(SAMPLE_FILE)))
    by_ticker = {r.ticker: r for r in rows}
    assert by_ticker["GME"].si_shares == 10_000_000
    assert by_ticker["GME"].settlement_date == date(2024, 4, 30)
    assert by_ticker["GME"].avg_daily_volume_20d == 2_000_000
    assert by_ticker["GME"].days_to_cover == 5.0


def test_finra_capabilities() -> None:
    p = FinraProvider()
    assert p.capabilities == frozenset({"si"})


@pytest.mark.asyncio
async def test_backfill_finra_requests_all_months_since_start(monkeypatch) -> None:
    """CDX-P2-5 regression: fetch_short_interest hardcoded range(0, 3) — only
    the last ~3 months of biweekly reports — while backfill_finra passes
    since=date(2018,1,1). A 2018→present backtest therefore got almost no
    short-interest history and f1/f2 were dead for years of data. The provider
    must request a report for EVERY month from `since` to today.
    """
    requested_yyyymm: set[str] = set()

    class _FakeResponse:
        text = ""

        def raise_for_status(self) -> None:
            # Pretend every month's file is missing so the loop just records
            # the attempt and continues (we only care which URLs were tried).
            raise httpx.HTTPStatusError("404", request=None, response=None)

    class _FakeClient:
        def __init__(self, *a, **kw) -> None:
            pass

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *a) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            # URL form: .../shrt{YYYYMM}{a|b}.txt — capture the YYYYMM.
            tail = url.rsplit("/", 1)[-1]  # shrt202401b.txt
            yyyymm = tail.removeprefix("shrt")[:6]
            requested_yyyymm.add(yyyymm)
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    p = FinraProvider()
    since = date(2018, 1, 1)
    await p.fetch_short_interest("GME", since=since)

    # Must have requested the earliest in-range month and span many months —
    # not just the trailing ~3. (Use a conservative lower bound so the test is
    # robust to the exact current date.)
    assert "201801" in requested_yyyymm, (
        f"earliest since-month 2018-01 never requested; got {sorted(requested_yyyymm)[:5]}..."
    )
    assert len(requested_yyyymm) >= 24, (
        f"only {len(requested_yyyymm)} months requested — still truncating history "
        f"instead of spanning since→today"
    )
