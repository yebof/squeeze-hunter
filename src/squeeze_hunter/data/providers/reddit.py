"""RedditProvider — PRAW wrapper for r/wallstreetbets mention counting."""

from __future__ import annotations

import asyncio
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
        # R8.Q-M2: case-insensitive so 'gme' matches GME on WSB (lowercase
        # ticker mentions are common). Prior bare-word pattern was the only
        # case-sensitive one in this list — inconsistent and undercounted.
        re.compile(rf"\b{ticker}\b", re.IGNORECASE),
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
        baseline = self._compute_baseline_30d(ticker, as_of)
        # R7.C9: when no real baseline is available (the cached daily-count
        # store is not yet implemented), return None so cross_sectional_z
        # treats f4 as missing for this ticker. Previously a hard-coded
        # (10, 20) prior was returned and applied to every ticker, every
        # day — masking a placeholder as a live signal with weight 1.5.
        if baseline is None:
            return None
        baseline_mean, baseline_std = baseline
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

        def _fetch_posts(sub_name: str) -> list:
            return list(self._reddit.subreddit(sub_name).new(limit=1000))

        total = 0
        for sub_name in self.subreddits:
            posts = await asyncio.to_thread(_fetch_posts, sub_name)
            posts = [p for p in posts if datetime.fromtimestamp(p.created_utc, tz=UTC) >= cutoff]
            total += _count_ticker_mentions(posts, ticker, aliases=[])
        return total

    def _compute_baseline_30d(
        self: RedditProvider, ticker: str, as_of: datetime
    ) -> tuple[float, float] | None:
        """R7.C9: returns None until the cached daily-count store exists.

        Caller treats None as "f4 unavailable for this ticker" and skips the
        factor row, so cross_sectional_z computes z over only the tickers
        with real data. Previously this returned a flat (10, 20) prior that
        was identical for every ticker — turning a 1.5-weighted live factor
        into noise-around-zero across the universe, with no warning.

        Re-introduction plan: nightly ingest writes per-day mention counts
        to `cache.write_partition('reddit_baseline', ticker, df)`. This
        method then reads the last 30 entries, computes mean/std, and
        returns those.
        """
        return None

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
