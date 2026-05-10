from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from squeeze_hunter.data.schema import (
    Bar,
    EarningsEvent,
    OptionChain,
    OptionQuote,
    Quote,
    RedditMention,
    ShortInterest,
)


def test_bar_validates_ohlc() -> None:
    Bar(
        ticker="GME",
        ts=datetime(2024, 5, 13, 13, 30, tzinfo=UTC),
        open=18.0,
        high=20.0,
        low=17.5,
        close=19.5,
        volume=1_000_000,
    )
    with pytest.raises(ValidationError):
        Bar(
            ticker="GME",
            ts=datetime(2024, 5, 13, 13, 30, tzinfo=UTC),
            open=18.0,
            high=15.0,
            low=17.5,
            close=19.5,
            volume=1_000_000,
        )


def test_short_interest_days_to_cover() -> None:
    si = ShortInterest(
        ticker="GME",
        settlement_date=date(2024, 4, 30),
        si_shares=1_000_000,
        si_pct_float=0.10,
        avg_daily_volume_20d=200_000,
    )
    assert si.days_to_cover == pytest.approx(5.0)


def test_option_chain_filter() -> None:
    chain = OptionChain(
        underlying="GME",
        as_of=datetime(2024, 5, 13, 20, 0, tzinfo=UTC),
        spot=18.0,
        quotes=[
            OptionQuote(
                strike=18.0,
                expiry=date(2024, 6, 21),
                right="C",
                open_interest=1000,
                volume=200,
                implied_vol=0.7,
                bid=1.0,
                ask=1.2,
            ),
            OptionQuote(
                strike=20.0,
                expiry=date(2024, 6, 21),
                right="C",
                open_interest=500,
                volume=100,
                implied_vol=0.8,
                bid=0.5,
                ask=0.6,
            ),
        ],
    )
    near = chain.near_money_calls(window_pct=0.05)
    assert len(near) == 1
    assert near[0].strike == 18.0


def test_reddit_mention_z_inputs() -> None:
    m = RedditMention(
        ticker="GME",
        as_of=datetime(2024, 5, 13, 12, 0, tzinfo=UTC),
        subreddit="wallstreetbets",
        count_24h=400,
        baseline_30d_mean=50.0,
        baseline_30d_std=20.0,
    )
    assert m.z_score == pytest.approx((400 - 50) / 20)


def test_earnings_event_surprise() -> None:
    e = EarningsEvent(
        ticker="GME",
        report_at=datetime(2024, 5, 12, 20, 30, tzinfo=UTC),
        actual_eps=0.10,
        estimate_eps=0.05,
    )
    assert e.surprise_pct == pytest.approx(1.0)


def test_quote_basic() -> None:
    Quote(
        ticker="GME",
        ts=datetime(2024, 5, 13, 14, 0, tzinfo=UTC),
        bid=18.0,
        ask=18.05,
        last=18.02,
        bid_size=100,
        ask_size=200,
    )
