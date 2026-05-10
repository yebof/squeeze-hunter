from datetime import date
from unittest.mock import patch

import pytest

from squeeze_hunter.data.providers.finnhub import FinnhubProvider


@pytest.mark.asyncio
async def test_finnhub_earnings_parses() -> None:
    fake = {
        "earningsCalendar": [
            {
                "symbol": "GME",
                "date": "2024-06-04",
                "hour": "amc",
                "epsActual": 0.10,
                "epsEstimate": 0.05,
                "revenueActual": 1_000_000_000,
                "revenueEstimate": 950_000_000,
            }
        ]
    }
    with patch.object(FinnhubProvider, "_call_calendar", return_value=fake):
        p = FinnhubProvider(api_key="x")
        events = await p.fetch_earnings("GME", since=date(2024, 1, 1))
    assert len(events) == 1
    assert events[0].ticker == "GME"
    assert events[0].surprise_pct == pytest.approx(1.0)


def test_finnhub_capabilities() -> None:
    p = FinnhubProvider(api_key="x")
    assert "earnings" in p.capabilities
