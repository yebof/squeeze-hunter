"""Round-12: f6 must sort bars; f6/f7 must not emit finite zeros for no-data tickers."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.data.schema import Bar
from squeeze_hunter.signals.technicals import compute_bollinger_breakout, compute_volume_spike

_CLOCK = datetime(2024, 5, 13, tzinfo=UTC)


def _bars(closes: list[float], volume: int = 1_000_000) -> list[Bar]:
    base = _CLOCK - timedelta(days=len(closes))
    return [
        Bar(
            ticker="GME", ts=base + timedelta(days=i), open=c, high=c, low=c, close=c, volume=volume
        )
        for i, c in enumerate(closes)
    ]


@pytest.mark.asyncio
async def test_bollinger_breakout_is_invariant_to_bar_order() -> None:
    bars = _bars([10.0] * 30 + [10.05] * 5 + [12.0])
    sorted_provider = AsyncMock()
    sorted_provider.fetch_bars.return_value = list(bars)
    reversed_provider = AsyncMock()
    reversed_provider.fetch_bars.return_value = list(reversed(bars))

    ordered = await compute_bollinger_breakout(["GME"], sorted_provider, _CLOCK)
    scrambled = await compute_bollinger_breakout(["GME"], reversed_provider, _CLOCK)

    a = float(ordered.values.set_index("ticker").loc["GME", "raw_value"])
    b = float(scrambled.values.set_index("ticker").loc["GME", "raw_value"])
    assert a > 0
    assert b == pytest.approx(a)


@pytest.mark.asyncio
async def test_technicals_emit_nan_not_zero_for_tickers_without_bars() -> None:
    """A finite 0.0 row enters the cross-sectional mean/std and shifts every
    other ticker's z. Missing data must be NaN (excluded), matching f1/f4."""
    bars = _bars([10.0] * 30 + [12.0])

    async def fetch_bars(ticker: str, *args, **kwargs) -> list[Bar]:
        return bars if ticker == "GME" else []

    provider = AsyncMock()
    provider.fetch_bars = AsyncMock(side_effect=fetch_bars)

    f6 = (await compute_bollinger_breakout(["GME", "NODATA"], provider, _CLOCK)).values
    f7 = (await compute_volume_spike(["GME", "NODATA"], provider, _CLOCK)).values
    for frame in (f6, f7):
        by_ticker = frame.set_index("ticker")["raw_value"]
        assert math.isnan(float(by_ticker["NODATA"]))
        assert math.isfinite(float(by_ticker["GME"]))
