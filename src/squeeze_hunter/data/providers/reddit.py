"""RedditProvider — PRAW wrapper for r/wallstreetbets mention counting."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import praw

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    Quote,
    RedditMention,
    ShortInterest,
)


def _count_ticker_mentions(posts: Iterable[object], ticker: str, aliases: list[str]) -> int:
    patterns = [
        re.compile(rf"\${ticker}\b", re.IGNORECASE),
        re.compile(rf"\b{ticker}\b"),
    ]
    for a in aliases:
        patterns.append(re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE))
    n = 0
    for post in posts:
        text = f"{getattr(post, 'title', '')} {getattr(post, 'selftext', '')}"
        if any(p.search(text) for p in patterns):
            n += 1
    return n


@dataclass
class RedditProvider:
    client_id: str
    client_secret: str
    user_agent: str
    subreddits: tuple[str, ...] = ("wallstreetbets",)
    name: str = "reddit"
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({"sentiment"}))

    def __post_init__(self: RedditProvider) -> None:
        self._reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            ratelimit_seconds=60,
        )

    async def fetch_sentiment(
        self: RedditProvider, ticker: str, as_of: datetime
    ) -> RedditMention | None:
        count_24h = await self._count_24h_mentions(ticker, as_of)
        baseline_mean, baseline_std = await self._compute_baseline_30d(ticker, as_of)
        return RedditMention(
            ticker=ticker,
            as_of=as_of,
            subreddit=self.subreddits[0],
            count_24h=count_24h,
            baseline_30d_mean=baseline_mean,
            baseline_30d_std=baseline_std,
        )

    async def _count_24h_mentions(self: RedditProvider, ticker: str, as_of: datetime) -> int:
        cutoff = as_of - timedelta(days=1)
        total = 0
        for sub_name in self.subreddits:
            sub = self._reddit.subreddit(sub_name)
            posts = list(sub.new(limit=1000))
            posts = [p for p in posts if datetime.fromtimestamp(p.created_utc, tz=UTC) >= cutoff]
            total += _count_ticker_mentions(posts, ticker, aliases=[])
        return total

    async def _compute_baseline_30d(
        self: RedditProvider, ticker: str, as_of: datetime
    ) -> tuple[float, float]:
        # Cheap baseline: per-day counts over the last 30 days, taken from search.
        # In Phase 1 we approximate from cached daily counts (recorded by nightly job).
        # Without history yet, fall back to a wide flat prior (mean=10, std=20) to
        # give well-behaved z-scores until enough days accumulate.
        return 10.0, 20.0

    # -- protocol stubs --
    async def fetch_bars(self: RedditProvider, *_a: object, **_kw: object) -> list[Bar]:
        return []

    async def fetch_quote(self: RedditProvider, ticker: str) -> Quote:
        raise NotImplementedError

    async def fetch_option_chain(self: RedditProvider, *_a: object, **_kw: object) -> OptionChain:
        return OptionChain(underlying="", as_of=datetime.now(UTC), spot=0.0, quotes=[])

    async def fetch_short_interest(
        self: RedditProvider, *_a: object, **_kw: object
    ) -> list[ShortInterest]:
        return []

    async def fetch_earnings(
        self: RedditProvider, *_a: object, **_kw: object
    ) -> list[EarningsEvent]:
        return []

    async def fetch_option_chain_at(
        self: RedditProvider, ticker: str, as_of: datetime
    ) -> OptionChain:
        """Live providers don't yet cache historical option chains. Returns empty."""
        return OptionChain(underlying=ticker, as_of=as_of, spot=0.0, quotes=[])
