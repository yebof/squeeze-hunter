from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from squeeze_hunter.signals.base import Factor
from squeeze_hunter.signals.compute import FACTOR_NAMES, compute_all_factors


@pytest.mark.asyncio
async def test_compute_all_returns_one_row_per_factor_per_ticker() -> None:
    tickers = ["GME", "AAPL"]

    def fake_factor(name: str) -> Factor:
        return Factor(
            name=name,
            as_of=datetime(2024, 5, 13, tzinfo=UTC),
            values=pd.DataFrame({"ticker": tickers, "raw_value": [1.0, 2.0]}),
        )

    async def stub_si(_t, _p, _c):
        return fake_factor("f1_si_pct")

    async def stub_dtc(_t, _p, _c):
        return fake_factor("f2_days_to_cover")

    async def stub_er(_t, _p, _c):
        return fake_factor("f3_earnings_reaction")

    async def stub_wsb(_t, _p, _c):
        return fake_factor("f4_wsb_mention")

    async def stub_oi(_t, _p, _c, **kw):
        return fake_factor("f5_call_oi_velocity")

    async def stub_bb(_t, _p, _c):
        return fake_factor("f6_bollinger_breakout")

    async def stub_vs(_t, _p, _c):
        return fake_factor("f7_volume_spike")

    provider = AsyncMock()
    with (
        patch("squeeze_hunter.signals.compute.compute_si_pct_float", stub_si),
        patch("squeeze_hunter.signals.compute.compute_days_to_cover", stub_dtc),
        patch("squeeze_hunter.signals.compute.compute_earnings_reaction", stub_er),
        patch("squeeze_hunter.signals.compute.compute_wsb_sentiment", stub_wsb),
        patch("squeeze_hunter.signals.compute.compute_call_oi_velocity", stub_oi),
        patch("squeeze_hunter.signals.compute.compute_bollinger_breakout", stub_bb),
        patch("squeeze_hunter.signals.compute.compute_volume_spike", stub_vs),
    ):
        df = await compute_all_factors(tickers, provider, datetime(2024, 5, 13, tzinfo=UTC))
    # Long format
    assert set(df.columns) >= {"ticker", "factor_name", "raw_value", "z_score"}
    assert set(df["factor_name"]) == set(FACTOR_NAMES)
    assert set(df["ticker"]) == {"GME", "AAPL"}
