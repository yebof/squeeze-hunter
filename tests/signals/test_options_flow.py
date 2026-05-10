from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.data.schema import OptionChain, OptionQuote
from squeeze_hunter.signals.options_flow import compute_call_oi_velocity


def _chain(spot: float, oi: int, ts: datetime) -> OptionChain:
    return OptionChain(
        underlying="GME",
        as_of=ts,
        spot=spot,
        quotes=[
            OptionQuote(
                strike=spot,
                expiry=date(2024, 6, 21),
                right="C",
                open_interest=oi,
                volume=10,
                implied_vol=0.7,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_call_oi_velocity_positive_when_growing() -> None:
    now = datetime(2024, 5, 13, 13, 30, tzinfo=UTC)
    week_ago = now - timedelta(days=7)
    provider = AsyncMock()

    async def chain(_t, expiry=None) -> OptionChain:
        return _chain(spot=18.0, oi=10000, ts=now)

    async def bars(t, start, end, resolution="1d"):
        return []

    provider.fetch_option_chain.side_effect = chain

    archive = {("GME", week_ago.date()): _chain(spot=17.0, oi=4000, ts=week_ago)}
    factor = await compute_call_oi_velocity(["GME"], provider, now, _archive=archive)
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] > 0  # OI grew 4k → 10k


@pytest.mark.asyncio
async def test_call_oi_velocity_zero_when_no_history() -> None:
    now = datetime(2024, 5, 13, 13, 30, tzinfo=UTC)
    week_ago = now - timedelta(days=7)
    provider = AsyncMock()

    async def chain(_t, expiry=None) -> OptionChain:
        return _chain(spot=18.0, oi=10000, ts=now)

    async def chain_at(_t, as_of: datetime) -> OptionChain:
        # No prior data available — return empty chain
        return OptionChain(underlying="GME", as_of=week_ago, spot=18.0, quotes=[])

    provider.fetch_option_chain.side_effect = chain
    provider.fetch_option_chain_at = chain_at
    # Pass explicit empty archive so provider lookup is still attempted (and returns empty)
    factor = await compute_call_oi_velocity(["GME"], provider, now, _archive={})
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] == 0.0


@pytest.mark.asyncio
async def test_call_oi_velocity_reads_prior_chain_from_provider() -> None:
    """C6 fix: when _archive is None, signal must query provider.fetch_option_chain_at
    for the 7-day-prior chain. Previously the signal silently returned 0.
    """
    now = datetime(2024, 5, 13, 13, 30, tzinfo=UTC)
    week_ago = now - timedelta(days=7)
    provider = AsyncMock()

    async def chain_today(_t, expiry=None) -> OptionChain:
        return _chain(spot=18.0, oi=10000, ts=now)

    async def chain_at(_t, as_of: datetime) -> OptionChain | None:
        # Provider serves the 7-day-prior snapshot
        if as_of.date() == week_ago.date():
            return _chain(spot=17.0, oi=4000, ts=week_ago)
        return None

    provider.fetch_option_chain.side_effect = chain_today
    provider.fetch_option_chain_at = chain_at  # AsyncMock auto-treats as coroutine

    factor = await compute_call_oi_velocity(["GME"], provider, now)
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] > 0  # OI grew 4k → 10k
