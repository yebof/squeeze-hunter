"""Round-12: `ingest finra` must not exit 0 when every FINRA download failed.

Observed live: cdn.finra.org answered 403 for all 210 monthly files, the
provider swallowed each HTTPStatusError, the backfill wrote nothing and the
CLI exited 0 — f1/f2 silently dead.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from squeeze_hunter.data.providers.finra import FinraProvider


class _Forbidden:
    text = ""
    status_code = 403

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError("403", request=None, response=None)  # type: ignore[arg-type]


class _FakeClient:
    def __init__(self, *a, **kw) -> None:
        pass

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *a) -> None:
        return None

    async def get(self, url: str) -> _Forbidden:
        return _Forbidden()


@pytest.mark.asyncio
async def test_bulk_fetch_raises_when_no_monthly_file_downloads(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    with pytest.raises(RuntimeError, match=r"0 of \d+ FINRA"):
        await FinraProvider().fetch_short_interest_bulk(["GME"], since=date(2024, 1, 1))
