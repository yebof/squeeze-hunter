"""NYSE trading calendar and session-time helpers — the single source of truth.

P5 of the architecture-hardening plan: the holiday list used to live in
`signals/earnings_reaction.py`, the session window in `runtime.py`, and the
backtest day loop rebuilt its own trading-day list. Every consumer (backtest
day loop, live time stop, f3 / f5 windows, FINRA publication lag,
captured-events) now imports from here.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

NY = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)

# Cover the realistic backtest + live range.
_RANGE_START = "2018-01-01"
_RANGE_END = "2030-12-31"

_HOLIDAYS: list[date] | None = None
_HOLIDAYS_DT64: np.ndarray | None = None


def nyse_holidays() -> list[date]:
    """Weekday NYSE closures 2018-2030: regular holidays plus special closures.

    Round-13: the previous US federal calendar iterated Good Friday (NYSE
    closed) and skipped Columbus Day and Veterans Day (NYSE open).
    """
    global _HOLIDAYS
    if _HOLIDAYS is None:
        nyse = mcal.get_calendar("NYSE")
        sessions = {ts.date() for ts in nyse.schedule(_RANGE_START, _RANGE_END).index}
        weekdays = pd.bdate_range(_RANGE_START, _RANGE_END)
        _HOLIDAYS = [ts.date() for ts in weekdays if ts.date() not in sessions]
    return _HOLIDAYS


def nyse_holidays_dt64() -> np.ndarray:
    """The same closures as a numpy `datetime64[D]` array for `np.busday_*`."""
    global _HOLIDAYS_DT64
    if _HOLIDAYS_DT64 is None:
        _HOLIDAYS_DT64 = np.array(nyse_holidays(), dtype="datetime64[D]")
    return _HOLIDAYS_DT64


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _holiday_set()


def next_session(d: date) -> date:
    """The first NYSE session strictly after `d`."""
    out = np.busday_offset(np.datetime64(d), 1, roll="forward", holidays=nyse_holidays_dt64())
    return out.astype("datetime64[D]").astype(object)


def trading_sessions(start: datetime, end: datetime) -> list[pd.Timestamp]:
    """UTC-midnight timestamps of every NYSE session in [start, end]."""
    holidays = _holiday_set()
    return [d for d in pd.bdate_range(start, end, tz="UTC") if d.date() not in holidays]


def is_regular_session(now: datetime) -> bool:
    """True if `now` falls within Mon-Fri 09:30-16:00 ET (regular trading).

    R3.2 kept this deliberately holiday-blind: trading on a holiday costs a
    logged skip, silently missing a half-day session would cost real money.
    """
    et = now.astimezone(NY)
    if et.weekday() >= 5:
        return False
    return SESSION_OPEN <= et.time() < SESSION_CLOSE


def session_open_utc(now: datetime) -> datetime:
    """Today's 09:30 ET (for `now`'s ET calendar day) expressed in UTC."""
    et = now.astimezone(NY)
    return datetime.combine(et.date(), SESSION_OPEN, tzinfo=NY).astimezone(UTC)


_HOLIDAY_SET: set[date] | None = None


def _holiday_set() -> set[date]:
    global _HOLIDAY_SET
    if _HOLIDAY_SET is None:
        _HOLIDAY_SET = set(nyse_holidays())
    return _HOLIDAY_SET
