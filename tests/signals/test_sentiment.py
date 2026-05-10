from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from squeeze_hunter.data.schema import RedditMention
from squeeze_hunter.signals.sentiment import compute_wsb_sentiment


@pytest.mark.asyncio
async def test_wsb_sentiment_returns_z_from_provider() -> None:
    provider = AsyncMock()
    provider.fetch_sentiment.side_effect = lambda t, ts: RedditMention(
        ticker=t,
        as_of=ts,
        subreddit="wallstreetbets",
        count_24h={"GME": 400, "AAPL": 8}[t],
        baseline_30d_mean=10.0,
        baseline_30d_std=20.0,
    )
    factor = await compute_wsb_sentiment(
        ["GME", "AAPL"], provider, datetime(2024, 5, 13, tzinfo=UTC)
    )
    out = factor.values.set_index("ticker")
    assert out.loc["GME", "raw_value"] == pytest.approx((400 - 10) / 20)
    assert out.loc["AAPL", "raw_value"] == pytest.approx((8 - 10) / 20)
