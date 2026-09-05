"""Round-13: the trading calendar must be NYSE, not the US federal calendar.

The federal calendar iterated Good Friday (NYSE closed → an equity point with
no marks, every year) and skipped Columbus Day and Veterans Day (NYSE open →
never scanned, never stop-checked, absent from the equity curve).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from squeeze_hunter.signals.earnings_reaction import _trading_days_between, _us_business_holidays


def test_calendar_matches_nyse_closures() -> None:
    holidays = set(_us_business_holidays())
    assert date(2024, 3, 29) in holidays  # Good Friday
    assert date(2025, 4, 18) in holidays  # Good Friday
    assert date(2025, 1, 9) in holidays  # National day of mourning (Carter)
    assert date(2024, 6, 19) in holidays  # Juneteenth
    assert date(2024, 10, 14) not in holidays  # Columbus Day: NYSE open
    assert date(2024, 11, 11) not in holidays  # Veterans Day: NYSE open
    assert date(2025, 11, 11) not in holidays


def test_trading_days_between_skips_good_friday() -> None:
    thu = datetime(2024, 3, 28, tzinfo=UTC)
    mon = datetime(2024, 4, 1, tzinfo=UTC)
    assert _trading_days_between(thu, mon) == 1
