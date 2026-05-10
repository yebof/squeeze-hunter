from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.data.schema import Bar
from squeeze_hunter.signals.technicals import (
    compute_bollinger_breakout,
    compute_volume_spike,
)


def _bars(closes: list[float], volumes: list[int]) -> list[Bar]:
    base = datetime(2024, 5, 13, tzinfo=UTC) - timedelta(days=len(closes))
    return [
        Bar(ticker="GME", ts=base + timedelta(days=i), open=c, high=c, low=c, close=c, volume=v)
        for i, (c, v) in enumerate(zip(closes, volumes, strict=False))
    ]


@pytest.mark.asyncio
async def test_bollinger_breakout_positive_after_squeeze_then_pop() -> None:
    flat = [10.0] * 30 + [10.05] * 5  # tight squeeze
    pop = [12.0]  # breakout
    bars = _bars(flat + pop, [1_000_000] * (len(flat) + len(pop)))
    provider = AsyncMock()
    provider.fetch_bars.return_value = bars
    factor = await compute_bollinger_breakout(["GME"], provider, datetime(2024, 5, 13, tzinfo=UTC))
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] > 0


@pytest.mark.asyncio
async def test_volume_spike_uses_today_vs_adv20() -> None:
    closes = [10.0] * 21
    volumes = [1_000_000] * 20 + [5_000_000]
    bars = _bars(closes, volumes)
    provider = AsyncMock()
    provider.fetch_bars.return_value = bars
    factor = await compute_volume_spike(["GME"], provider, datetime(2024, 5, 13, tzinfo=UTC))
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] == pytest.approx(5.0)
