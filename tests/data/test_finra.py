from datetime import date
from io import StringIO

import pytest

from squeeze_hunter.data.providers.finra import FinraProvider

SAMPLE_FILE = """Settlement Date|Symbol Code|Symbol|Market Class Code|Current Short Position|Previous Short Position|Change|Percent Change|Average Daily Volume|Days To Cover|Revision Indicator
20240430|GME|GME|N|10000000|9500000|500000|5.26|2000000|5.0|N
20240430|AAPL|AAPL|N|50000000|48000000|2000000|4.17|80000000|0.625|N
"""


@pytest.mark.asyncio
async def test_parse_finra_sample() -> None:
    p = FinraProvider()
    rows = list(p._parse_finra_pipe_text(StringIO(SAMPLE_FILE)))
    by_ticker = {r.ticker: r for r in rows}
    assert by_ticker["GME"].si_shares == 10_000_000
    assert by_ticker["GME"].settlement_date == date(2024, 4, 30)
    assert by_ticker["GME"].avg_daily_volume_20d == 2_000_000
    assert by_ticker["GME"].days_to_cover == 5.0


def test_finra_capabilities() -> None:
    p = FinraProvider()
    assert p.capabilities == frozenset({"si"})
