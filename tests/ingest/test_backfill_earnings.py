"""Round-12: `ingest earnings` must not silently no-op without an API key."""

from __future__ import annotations

from pathlib import Path

import pytest

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.ingest.backfill_earnings import backfill_earnings


@pytest.mark.asyncio
async def test_backfill_earnings_refuses_to_run_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FINNHUB_KEY", raising=False)
    with pytest.raises(ValueError, match="FINNHUB_KEY"):
        await backfill_earnings(["GME"], ParquetCache(root=tmp_path))
    assert not (tmp_path / "earnings").exists()
