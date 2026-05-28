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


@pytest.mark.asyncio
async def test_finnhub_bmo_stamps_report_at_before_report_day_bar() -> None:
    """Round-11 regression: only `amc` was special-cased; `bmo` (before market
    open), `dmh` (during market hours), and unknown sessions defaulted to 12:30
    UTC — AFTER the report-day daily bar (yfinance daily bars are ET-midnight →
    04:00-05:00 UTC). So a before-open reaction got classified as PRE-event and
    f3 (earnings reaction, weight 2.0, primary CAR driver) measured the wrong
    (next) day. BMO/DMH must stamp report_at BEFORE the report-day bar so the
    report-day close is the first post-event bar; AMC stays after the close.
    """
    fake = {
        "earningsCalendar": [
            {
                "symbol": "GME",
                "date": "2024-06-04",
                "hour": "bmo",
                "epsActual": 0.10,
                "epsEstimate": 0.05,
            },
            {
                "symbol": "GME",
                "date": "2024-07-10",
                "hour": "dmh",
                "epsActual": 0.10,
                "epsEstimate": 0.05,
            },
            {
                "symbol": "GME",
                "date": "2024-09-10",
                "hour": "amc",
                "epsActual": 0.10,
                "epsEstimate": 0.05,
            },
        ]
    }
    with patch.object(FinnhubProvider, "_call_calendar", return_value=fake):
        p = FinnhubProvider(api_key="x")
        events = await p.fetch_earnings("GME", since=date(2024, 1, 1))
    by_date = {e.report_at.date().isoformat(): e.report_at for e in events}
    # BMO and DMH must sort BEFORE the report-day daily bar (~04:00-05:00 UTC),
    # i.e. very early in the report-day UTC.
    assert by_date["2024-06-04"].hour < 4
    assert by_date["2024-07-10"].hour < 4
    # AMC stays after the close so the same-day bar correctly remains pre-event.
    assert by_date["2024-09-10"].hour == 20


def test_finnhub_capabilities() -> None:
    p = FinnhubProvider(api_key="x")
    assert "earnings" in p.capabilities
