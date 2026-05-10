from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
import pytest

from squeeze_hunter.data.providers.yahoo import YahooProvider


@pytest.mark.asyncio
async def test_yahoo_fetch_earnings_handles_pd_nat() -> None:
    """C2_minor: pd.notna check, not identity comparison."""
    from unittest.mock import MagicMock, patch

    fake_ticker = MagicMock()
    fake_ticker.calendar = {"Earnings Date": [pd.NaT], "Earnings Average": None}
    with patch("yfinance.Ticker", return_value=fake_ticker):
        p = YahooProvider()
        events = await p.fetch_earnings("GME")
    assert events == []


@pytest.mark.asyncio
async def test_yahoo_fetch_earnings_handles_tz_aware_timestamp() -> None:
    """R4 regression: modern yfinance returns tz-aware Timestamps for many tickers.
    Calling tz_localize on an aware Timestamp raises TypeError; the previous
    code crashed for any non-naive earnings date.
    """
    from unittest.mock import MagicMock, patch

    fake_ticker = MagicMock()
    fake_ticker.calendar = {
        "Earnings Date": [pd.Timestamp("2024-05-13 16:30", tz="US/Eastern")],
        "Earnings Average": 0.05,
    }
    with patch("yfinance.Ticker", return_value=fake_ticker):
        p = YahooProvider()
        events = await p.fetch_earnings("GME")
    assert len(events) == 1
    # tz_convert preserves the absolute instant; assert UTC equivalent of 16:30 ET
    # 16:30 ET in May (EDT, UTC-4) → 20:30 UTC
    assert events[0].report_at.hour == 20
    assert events[0].report_at.minute == 30
    assert events[0].report_at.tzinfo is not None


@pytest.mark.asyncio
async def test_yahoo_fetch_earnings_handles_naive_timestamp() -> None:
    """Naive timestamp should still work — tz_localize path."""
    from unittest.mock import MagicMock, patch

    fake_ticker = MagicMock()
    fake_ticker.calendar = {
        "Earnings Date": [pd.Timestamp("2024-05-13 12:00")],  # naive
        "Earnings Average": 0.05,
    }
    with patch("yfinance.Ticker", return_value=fake_ticker):
        p = YahooProvider()
        events = await p.fetch_earnings("GME")
    assert len(events) == 1
    assert events[0].report_at.tzinfo is not None


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
