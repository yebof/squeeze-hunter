from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.data.schema import Bar, EarningsEvent
from squeeze_hunter.signals.earnings_reaction import (
    _trading_days_between,
    compute_earnings_reaction,
)


def test_trading_days_between_excludes_us_federal_holidays() -> None:
    """R8 regression: Christmas Day (federal holiday) is no longer counted
    as a trading day. Mon→Fri across Christmas (Wed) should be 3 trading
    days, not 4. Without R8 the result was 4 (Mon, Tue, Christmas, Fri).
    """
    mon = datetime(2024, 12, 23, tzinfo=UTC)  # Mon before Christmas
    fri = datetime(2024, 12, 27, tzinfo=UTC)  # Fri after
    assert _trading_days_between(mon, fri) == 3


def test_trading_days_between_handles_thanksgiving_week() -> None:
    """Thanksgiving 2024 was Thu Nov 28. Mon->Fri (end-exclusive) spans
    Mon, Tue, Wed, Thu — but Thanksgiving Thu is now a holiday, so 3 trading
    days. Without R8 the result was 4."""
    mon = datetime(2024, 11, 25, tzinfo=UTC)
    fri = datetime(2024, 11, 29, tzinfo=UTC)
    assert _trading_days_between(mon, fri) == 3


def test_trading_days_between_normal_week_unchanged() -> None:
    """A normal Mon->Fri (no holidays) should still be 4 trading days."""
    mon = datetime(2024, 5, 13, tzinfo=UTC)  # no holiday this week
    fri = datetime(2024, 5, 17, tzinfo=UTC)
    assert _trading_days_between(mon, fri) == 4


def _bar(ticker: str, ts: datetime, close: float, volume: int) -> Bar:
    return Bar(ticker=ticker, ts=ts, open=close, high=close, low=close, close=close, volume=volume)


@pytest.mark.asyncio
async def test_earnings_reaction_uses_trading_days_for_window() -> None:
    """B4: the 5-day decay window must be in trading days, not calendar days.

    Scenario: report is pre-market Friday (08:00 UTC), clock is Thursday evening
    (20:30 UTC) the following week.  That is 6 calendar days (.days == 6) but only
    4 trading days (Mon, Tue, Wed, Thu).  The old calendar-day filter saw 6 > 5 and
    dropped the event; the corrected trading-day filter keeps it and decay=0.2>0.
    """
    earnings_friday = datetime(2024, 5, 10, 8, 0, tzinfo=UTC)  # Friday pre-market
    later_thursday = datetime(
        2024, 5, 16, 20, 30, tzinfo=UTC
    )  # Thursday evening (6 cal / 4 trading days later)
    pre_bars = [
        _bar("GME", earnings_friday - timedelta(days=d), 18.0, 1_000_000) for d in range(20, 0, -1)
    ]
    post_bar = _bar("GME", earnings_friday + timedelta(days=1), 22.0, 8_000_000)
    provider = AsyncMock()
    provider.fetch_earnings.return_value = [
        EarningsEvent(ticker="GME", report_at=earnings_friday, actual_eps=0.10, estimate_eps=0.05)
    ]
    provider.fetch_bars.return_value = [*pre_bars, post_bar]
    factor = await compute_earnings_reaction(["GME"], provider, later_thursday)
    out = factor.values.set_index("ticker")
    # 4 trading days later, decay=0.2 → nonzero raw_value
    # (with the old calendar-day filter: 6 cal-days > 5 → would be dropped, raw=0)
    assert out.loc["GME", "raw_value"] != 0.0


@pytest.mark.asyncio
async def test_earnings_reaction_uses_post_event_gap_and_volume() -> None:
    earnings_at = datetime(2024, 5, 12, 20, 30, tzinfo=UTC)  # AMC, 2024-05-12
    pre_bars = [
        _bar("GME", earnings_at - timedelta(days=d), 18.0, 1_000_000) for d in range(20, 0, -1)
    ]
    post_bar = _bar("GME", earnings_at + timedelta(days=1), 22.0, 8_000_000)
    provider = AsyncMock()
    provider.fetch_earnings.return_value = [
        EarningsEvent(ticker="GME", report_at=earnings_at, actual_eps=0.10, estimate_eps=0.05)
    ]
    provider.fetch_bars.return_value = [*pre_bars, post_bar]
    factor = await compute_earnings_reaction(
        ["GME"], provider, datetime(2024, 5, 13, 13, 0, tzinfo=UTC)
    )
    out = factor.values.set_index("ticker")
    # gap ~22% with volume ratio 8x → strong positive raw value
    assert out.loc["GME", "raw_value"] > 0
