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


@pytest.mark.asyncio
async def test_yahoo_get_float_shares() -> None:
    from unittest.mock import MagicMock, patch

    fake_ticker = MagicMock()
    fake_ticker.info = {"floatShares": 12345678, "sharesOutstanding": 99999999}
    with patch("yfinance.Ticker", return_value=fake_ticker):
        p = YahooProvider()
        n = await p.get_float_shares("GME")
    assert n == 12345678


@pytest.mark.asyncio
async def test_yahoo_get_float_shares_falls_back_to_shares_outstanding() -> None:
    from unittest.mock import MagicMock, patch

    fake_ticker = MagicMock()
    fake_ticker.info = {"sharesOutstanding": 50000000}  # no floatShares
    with patch("yfinance.Ticker", return_value=fake_ticker):
        p = YahooProvider()
        n = await p.get_float_shares("GME")
    assert n == 50000000


@pytest.mark.asyncio
async def test_yahoo_get_float_shares_returns_none_when_missing() -> None:
    from unittest.mock import MagicMock, patch

    fake_ticker = MagicMock()
    fake_ticker.info = {}
    with patch("yfinance.Ticker", return_value=fake_ticker):
        p = YahooProvider()
        n = await p.get_float_shares("UNKNOWN")
    assert n is None
