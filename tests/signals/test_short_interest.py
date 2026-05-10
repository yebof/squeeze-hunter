from datetime import UTC, date, datetime
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.data.schema import ShortInterest
from squeeze_hunter.signals.short_interest import compute_days_to_cover, compute_si_pct_float


@pytest.mark.asyncio
async def test_si_pct_float_factor_picks_latest() -> None:
    provider = AsyncMock()
    provider.fetch_short_interest.side_effect = lambda t, since=None: [
        ShortInterest(
            ticker=t,
            settlement_date=date(2024, 4, 30),
            si_shares={"GME": 5_000_000, "AAPL": 1_000_000}[t],
            si_pct_float={"GME": 0.20, "AAPL": 0.01}[t],
            avg_daily_volume_20d={"GME": 1_000_000, "AAPL": 80_000_000}[t],
        )
    ]
    factor = await compute_si_pct_float(
        ["GME", "AAPL"], provider, datetime(2024, 5, 13, tzinfo=UTC)
    )
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] == 0.20
    assert out.loc["AAPL", "raw_value"] == 0.01


@pytest.mark.asyncio
async def test_si_pct_float_excludes_tickers_with_zero_float() -> None:
    """A ticker with si_pct_float=0 (no float data) must not be emitted as a 0 row,
    or it pollutes cross-sectional z-score for other tickers.
    """
    provider = AsyncMock()

    def make_si(t: str, pct: float) -> list[ShortInterest]:
        return [
            ShortInterest(
                ticker=t,
                settlement_date=date(2024, 4, 30),
                si_shares=1_000_000,
                si_pct_float=pct,
                avg_daily_volume_20d=200_000,
            )
        ]

    provider.fetch_short_interest.side_effect = lambda t, since=None: make_si(
        t, {"GME": 0.20, "AAPL": 0.0, "TSLA": 0.05}[t]
    )
    factor = await compute_si_pct_float(
        ["GME", "AAPL", "TSLA"], provider, datetime(2024, 5, 13, tzinfo=UTC)
    )
    tickers = set(factor.values["ticker"])
    assert "GME" in tickers
    assert "TSLA" in tickers
    assert "AAPL" not in tickers  # excluded — si_pct_float=0 means no float data


@pytest.mark.asyncio
async def test_days_to_cover_uses_si_div_adv() -> None:
    provider = AsyncMock()
    provider.fetch_short_interest.return_value = [
        ShortInterest(
            ticker="GME",
            settlement_date=date(2024, 4, 30),
            si_shares=5_000_000,
            si_pct_float=0.20,
            avg_daily_volume_20d=1_000_000,
        )
    ]
    factor = await compute_days_to_cover(["GME"], provider, datetime(2024, 5, 13, tzinfo=UTC))
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] == pytest.approx(5.0)
