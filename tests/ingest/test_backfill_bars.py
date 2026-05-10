from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.ingest.backfill_bars import backfill_bars_for_ticker


@pytest.mark.asyncio
async def test_backfill_writes_partition(tmp_path: Path) -> None:
    fake = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.5],
            "Close": [10.5, 11.5],
            "Volume": [1_000_000, 1_500_000],
        },
        index=pd.to_datetime(["2024-05-10", "2024-05-13"], utc=True),
    )
    cache = ParquetCache(root=tmp_path)
    with patch("squeeze_hunter.data.providers.yahoo._yf_history", return_value=fake):
        await backfill_bars_for_ticker(
            "GME",
            datetime(2024, 5, 1, tzinfo=UTC),
            datetime(2024, 5, 14, tzinfo=UTC),
            cache=cache,
        )
    out = cache.read_partition("bars", "GME")
    assert len(out) == 2
