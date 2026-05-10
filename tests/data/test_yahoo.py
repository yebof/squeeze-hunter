from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
import pytest

from squeeze_hunter.data.providers.yahoo import YahooProvider


@pytest.mark.asyncio
async def test_yahoo_fetch_bars_normalizes() -> None:
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
    with patch("squeeze_hunter.data.providers.yahoo._yf_history", return_value=fake):
        p = YahooProvider()
        bars = await p.fetch_bars(
            "GME",
            datetime(2024, 5, 10, tzinfo=UTC),
            datetime(2024, 5, 14, tzinfo=UTC),
        )
    assert len(bars) == 2
    assert bars[0].ticker == "GME"
    assert bars[0].close == 10.5


def test_yahoo_capabilities() -> None:
    p = YahooProvider()
    assert "bars" in p.capabilities
    assert "options" in p.capabilities
