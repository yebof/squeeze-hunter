from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

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

    # A2: _compute_baseline_30d is now a regular (non-async) function
    monkeypatch.setattr(p, "_count_24h_mentions", fake_count_recent)
    monkeypatch.setattr(p, "_compute_baseline_30d", lambda *a, **kw: (50.0, 20.0))
    m = await p.fetch_sentiment("GME", datetime(2024, 5, 13, tzinfo=UTC))
    assert m is not None
    assert m.count_24h == 400
    assert m.z_score == pytest.approx((400 - 50) / 20)


@pytest.mark.asyncio
async def test_reddit_count_uses_to_thread() -> None:
    """A1 regression: PRAW pagination must run via asyncio.to_thread, not block the loop.

    We verify by patching asyncio.to_thread and asserting it is called.
    """
    from unittest.mock import AsyncMock, patch

    p = RedditProvider(client_id="x", client_secret="y", user_agent="t")
    # Replace praw with a lightweight stub
    p._reddit = MagicMock()
    fake_subreddit = MagicMock()
    fake_subreddit.new = MagicMock(return_value=[])
    p._reddit.subreddit = MagicMock(return_value=fake_subreddit)

    with patch(
        "squeeze_hunter.data.providers.reddit.asyncio.to_thread", new=AsyncMock(return_value=[])
    ) as m:
        await p._count_24h_mentions("GME", datetime(2024, 5, 13, tzinfo=UTC))
    assert m.called
