from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.data.schema import Bar, EarningsEvent
from squeeze_hunter.signals.earnings_reaction import compute_earnings_reaction


def _bar(ticker: str, ts: datetime, close: float, volume: int) -> Bar:
    return Bar(ticker=ticker, ts=ts, open=close, high=close, low=close, close=close, volume=volume)


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
