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


@pytest.mark.asyncio
async def test_compute_all_factors_propagates_programming_errors() -> None:
    """CDX-P2-4 regression: gather(return_exceptions=True) + a blanket
    `isinstance(result, BaseException)` skip swallowed AttributeError /
    TypeError / NotImplementedError / CancelledError — exactly the
    programming/contract errors CLAUDE.md says MUST propagate to tick_safe.
    A real bug (renamed attr, wrong signature) would silently degrade to a
    smaller factor set and a quietly-wrong score instead of failing loud.
    """

    async def ok_factor(_t, _p, _c):
        return Factor(
            name="f1_si_pct",
            as_of=datetime(2024, 5, 13, tzinfo=UTC),
            values=pd.DataFrame({"ticker": ["GME"], "raw_value": [1.0]}),
        )

    async def buggy_factor(_t, _p, _c):
        raise AttributeError("provider has no attribute 'fetch_xyz' (real bug)")

    provider = AsyncMock()
    with (
        patch("squeeze_hunter.signals.compute.compute_si_pct_float", ok_factor),
        patch("squeeze_hunter.signals.compute.compute_days_to_cover", buggy_factor),
        patch("squeeze_hunter.signals.compute.compute_earnings_reaction", ok_factor),
        patch("squeeze_hunter.signals.compute.compute_wsb_sentiment", ok_factor),
        patch("squeeze_hunter.signals.compute.compute_call_oi_velocity", ok_factor),
        patch("squeeze_hunter.signals.compute.compute_bollinger_breakout", ok_factor),
        patch("squeeze_hunter.signals.compute.compute_volume_spike", ok_factor),
        pytest.raises(AttributeError, match="real bug"),
    ):
        await compute_all_factors(["GME"], provider, datetime(2024, 5, 13, tzinfo=UTC))


@pytest.mark.asyncio
async def test_compute_all_factors_still_skips_transient_errors() -> None:
    """CDX-P2-4: transient I/O errors (ConnectionError/TimeoutError/OSError)
    in ONE factor must still be logged-and-skipped — the scan degrades
    gracefully on a flaky network rather than aborting the whole run.
    """

    async def ok_factor(_t, _p, _c):
        return Factor(
            name="f1_si_pct",
            as_of=datetime(2024, 5, 13, tzinfo=UTC),
            values=pd.DataFrame({"ticker": ["GME"], "raw_value": [1.0]}),
        )

    async def flaky_factor(_t, _p, _c):
        raise ConnectionError("transient network blip")

    provider = AsyncMock()
    with (
        patch("squeeze_hunter.signals.compute.compute_si_pct_float", ok_factor),
        patch("squeeze_hunter.signals.compute.compute_days_to_cover", flaky_factor),
        patch("squeeze_hunter.signals.compute.compute_earnings_reaction", ok_factor),
        patch("squeeze_hunter.signals.compute.compute_wsb_sentiment", ok_factor),
        patch("squeeze_hunter.signals.compute.compute_call_oi_velocity", ok_factor),
        patch("squeeze_hunter.signals.compute.compute_bollinger_breakout", ok_factor),
        patch("squeeze_hunter.signals.compute.compute_volume_spike", ok_factor),
    ):
        df = await compute_all_factors(["GME"], provider, datetime(2024, 5, 13, tzinfo=UTC))
    # The flaky factor is dropped; the other 6 still produced rows.
    assert "f2_days_to_cover" not in set(df["factor_name"])
    assert "f1_si_pct" in set(df["factor_name"])
