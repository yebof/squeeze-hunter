from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from squeeze_hunter.data.providers.reddit import RedditProvider, _count_ticker_mentions


def test_mention_counter_matches_dollar_signs_and_words() -> None:
    posts = [
        SimpleNamespace(title="$GME to the moon", selftext="loaded calls"),
        SimpleNamespace(title="GameStop earnings tomorrow", selftext="$GME paper hands sell"),
        SimpleNamespace(title="AAPL is fine", selftext="boring"),
    ]
    n = _count_ticker_mentions(posts, ticker="GME", aliases=["GameStop"])
    assert n == 2  # third post has no GME / GameStop


@pytest.mark.asyncio
async def test_provider_z_score_uses_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    p = RedditProvider(client_id="x", client_secret="y", user_agent="t")

    async def fake_count_recent(*a, **kw) -> int:
        return 400

    async def fake_baseline(*a, **kw) -> tuple[float, float]:
        return 50.0, 20.0

    monkeypatch.setattr(p, "_count_24h_mentions", fake_count_recent)
    monkeypatch.setattr(p, "_compute_baseline_30d", fake_baseline)
    m = await p.fetch_sentiment("GME", datetime(2024, 5, 13, tzinfo=UTC))
    assert m is not None
    assert m.count_24h == 400
    assert m.z_score == pytest.approx((400 - 50) / 20)
